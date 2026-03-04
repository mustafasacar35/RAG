import os
import uuid
import httpx
import re
import io
import json
import asyncio
import time
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from google import genai
from google.genai import types

import PyPDF2
from docx import Document
import ebooklib
from ebooklib import epub
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ─── App & DB ────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kkzommsulkxccrcmhtnq.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtrem9tbXN1bGt4Y2NyY21odG5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI1NjU3MTQsImV4cCI6MjA4ODE0MTcxNH0.PROlh1JVQkylMcGuwyoBzxrE2z_o_dIYa_HGM8hAKHs")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Job store: crawl_jobs[job_id] = {...}
crawl_jobs: dict[str, dict] = {}

# ─── Config ──────────────────────────────────────────────────────────────────

GEMINI_API_KEY_ENV  = os.environ.get("GEMINI_API_KEY", "")

MAX_FILE_SIZE_MB    = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
CRAWL_DELAY_SEC     = 0.4          #礼儀 — sunucuyu zorlamayalım
PAGES_PER_LAYER     = 50           # Katman başına max sayfa
AUTO_SCORE_THRESHOLD = 0.45        # Bu skorun üstündeyse "yeterli" sayar (cosine distance, düşük = iyi)

# ─── Google Drive Config ─────────────────────────────────────────────────────
GOOGLE_DRIVE_FOLDER_ID      = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "1XkqOenGiKibLX54zKNWefxtGiWIioWTX")

# Credentials: env var (JSON string) veya dosya yolu
_sa_env = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
if _sa_env and os.path.isfile(_sa_env):
    GOOGLE_SERVICE_ACCOUNT_JSON = Path(_sa_env).read_text(encoding="utf-8")
elif _sa_env:
    GOOGLE_SERVICE_ACCOUNT_JSON = _sa_env
elif os.path.isfile("service_account.json"):
    GOOGLE_SERVICE_ACCOUNT_JSON = Path("service_account.json").read_text(encoding="utf-8")
else:
    GOOGLE_SERVICE_ACCOUNT_JSON = ""

# Drive sync state
drive_sync_status: dict = {
    "status": "idle",       # idle | running | done | error
    "last_sync": None,
    "files_synced": 0,
    "total_chunks": 0,
    "message": "",
    "files": [],            # [{name, chunks, drive_id}]
}

DEFAULT_SYSTEM_PROMPT = """Sen deneyimli bir sağlık danışmanısın. Sana verilen kaynak belgelerden yararlanarak soruları yanıtla.
Yanıtlarını YALNIZCA sağlanan kaynaklara dayandır. Kaynaklarda bilgi yoksa bunu belirt.
Türkçe sorulara Türkçe, İngilizce sorulara İngilizce yanıt ver.

KESİN FORMAT KURALLARI (bunlara kesinlikle uy, ihlal etme):
1. Yanıtlarında hangi bilgiyi hangi kaynaktan aldıysan cümlenin sonuna o kaynağın numarasını köşeli parantez içinde yaz. Örnek: "...olduğu bilinmektedir [1]." veya "...gösterilmiştir [2]."
2. SADECE metin içinde atıf numarası kullan ([1], [2] vb.). En alta kaynaklar listesi EKLEME. 
3. Asla yıldız karakteri kullanma. Ne tek yıldız ne çift yıldız. Hiçbir kelimeyi yıldızlarla çevreleme.
4. Asla markdown formatı kullanma. Başlık için # işareti koyma.
5. Asla numaralı veya madde işaretli liste yapma.
6. Asla tire (-) veya nokta (•) ile başlayan satırlar oluşturma.
7. Her şeyi düz paragraflar halinde yaz. Bilgileri akıcı cümlelerle birbirine bağla.
8. Bir doktorla sohbet eder gibi doğal ve samimi bir üslup kullan.
9. Kalın, italik veya herhangi bir metin vurgulaması yapma."""

def strip_markdown(text: str) -> str:
    """AI yanıtından kalan markdown işaretlerini temizler."""
    # **bold** ve *italic* kaldır
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # ### başlıklar kaldır
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # - veya * ile başlayan liste öğelerini düz metne çevir
    text = re.sub(r'^[\-\*•]\s+', '', text, flags=re.MULTILINE)
    # Numaralı liste (1. 2. vb) → düz metin
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def get_api_key(x: Optional[str] = None) -> str:
    return x or GEMINI_API_KEY_ENV or ""

# ─── Text & Embed Helpers ────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

async def get_embedding(text: str, api_key: str) -> list[float]:
    client = genai.Client(api_key=api_key.strip())
    
    try:
        # 1. Try text-embedding-004 with native 768 config
        response = await asyncio.to_thread(
            client.models.embed_content,
            model='text-embedding-004',
            contents=text[:8000],
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        vals = response.embeddings[0].values
    except Exception as e:
        logging.warning(f"text-embedding-004 target failed, falling back to older model: {e}")
        try:
            # 2. Fallback to older gemini model format
            response = await asyncio.to_thread(
                client.models.embed_content,
                model='models/embedding-001',
                contents=text[:8000]
            )
            vals = response.embeddings[0].values
        except Exception as fallback_e:
            logging.error(f"Embedding API Error:\n{fallback_e}")
            raise Exception(f"Tüm modeller başarısız oldu: {str(fallback_e)}")
            
    # Force strictly 768 dimensions (Truncate if 3072; Pad with 0s if short)
    if len(vals) > 768:
        vals = vals[:768]
    elif len(vals) < 768:
        vals = vals + [0.0] * (768 - len(vals))
        
    return vals

async def gemini_chat(prompt: str, system_prompt: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    response = await asyncio.to_thread(
        client.models.generate_content,
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            max_output_tokens=8192
        )
    )
    return response.text

async def index_chunks(chunks: list[str], source: str, meta: dict, api_key: str) -> int:
    doc_id = str(uuid.uuid4())[:12]
    
    # 1) Önce belgeyi/kaynağı temsil eden klasör/dosya mantığını birleştirip Supabase'e atmak gerek.
    # Geçici çözüm: Tüm belge satırlarını "documents" tablosuna ekle
    # 'meta' içindeki bilgileri ayrı bir jsonb kolonunda tutacağız.
    
    records = []
    for i, c in enumerate(chunks):
        emb = await get_embedding(c, api_key)
        chunk_id = f"{doc_id}_{i}"
        
        # 'folder_id' bilgisi burada meta içinde gelmiyorsa null olabilir.
        # Aslında folder_id'yi index_chunks'ı çağıran yerler ayarlıyor ama API'dan gelen klasöre göre.
        # Şimdilik basitçe tabloya yazalım, folder id'ler update_folders vs ile handle edilecek.
        
        records.append({
            "id": chunk_id,
            "source": source,
            "content": c,
            "embedding": emb,
            "metadata": {"chunk_index": i, **{k: str(v) for k, v in meta.items()}}
        })
        
    if records:
        # Supabase Python SDK ile toplu ekleme
        try:
            res = await asyncio.to_thread(lambda: supabase.table("documents").insert(records).execute())
        except Exception as e:
            logging.error(f"Supabase insert error: {e}")
            raise HTTPException(500, f"Veritabanı kayıt hatası (Supabase): {str(e)[:200]}")
            
    return len(records)

# ─── Auto-score: "bu konuda yeterli içerik var mı?" ──────────────────────────

async def content_score(query_hint: str, api_key: str) -> float:
    """
    Index'e bir skor sorgusu atar. (Supabase pgvector)
    En iyi eşleşmenin cosine distance'ını döner (0=mükemmel, 1=alakasız).
    Skor < threshold → "yeterli içerik bulundu".
    """
    if not query_hint:
        return 1.0
        
    try:
        count_res = await asyncio.to_thread(lambda: supabase.table("documents").select("id", count="exact").limit(1).execute())
        if not count_res.count or count_res.count == 0:
            return 1.0
            
        emb = await get_embedding(query_hint, api_key)
        emb_str = f"[{','.join(map(str, emb))}]"
        
        # SQL rpc call
        res = await asyncio.to_thread(
            lambda: supabase.rpc('match_documents', {
                'query_embedding': emb_str,
                'match_threshold': 0.0,
                'match_count': 1
            }).execute()
        )
        
        if res.data:
            # similarity returns 1 - cosine_distance. Thus distance = 1 - similarity.
            sim = res.data[0].get("similarity", 0)
            return 1.0 - sim
            
        return 1.0
    except Exception as e:
        logging.error(f"Supabase score error: {e}")
        return 1.0

# ─── File Extractors ─────────────────────────────────────────────────────────

def extract_pdf(data: bytes) -> str:
    return "\n".join(p.extract_text() or "" for p in PyPDF2.PdfReader(io.BytesIO(data)).pages)

def extract_docx_b(data: bytes) -> str:
    return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)

def extract_epub_b(data: bytes) -> str:
    book = epub.read_epub(io.BytesIO(data))
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for t in soup(["script", "style"]): t.decompose()
        parts.append(soup.get_text(" "))
    return " ".join(parts)

# ─── Google Drive Helpers ─────────────────────────────────────────────────────

DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
DRIVE_MIME_MAP = {
    'application/pdf': 'pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/epub+zip': 'epub',
    'text/plain': 'txt',
    'text/markdown': 'md',
    # Google Docs → export as plain text
    'application/vnd.google-apps.document': 'gdoc',
}

def get_drive_service():
    """Service Account ile Google Drive API client oluşturur."""
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON ayarlanmamış")
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_drive_files(service, folder_id: str) -> list[dict]:
    """Drive klasöründeki desteklenen dosyaları listeler."""
    query = f"'{folder_id}' in parents and trashed = false"
    results = []
    page_token = None
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
            pageSize=100,
            pageToken=page_token
        ).execute()
        for f in resp.get('files', []):
            mime = f.get('mimeType', '')
            if mime in DRIVE_MIME_MAP or mime.startswith('text/'):
                results.append(f)
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return results

def resolve_drive_folder_id(service, folder_input: str) -> str:
    """Kullanıcı klasör ismi girdiyse ID'sini bulur, ID girdiyse aynen döner."""
    folder_input = folder_input.strip()
    if not folder_input:
        return ""
    
    # ID format check (Google Drive IDs are usually 25+ chars, no spaces)
    if len(folder_input) > 20 and " " not in folder_input:
        return folder_input
        
    query = f"name='{folder_input}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    resp = service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
    files = resp.get('files', [])
    if not files:
        raise ValueError(f"Drive'da '{folder_input}' adında bir klasör bulunamadı.")
    return files[0]['id']

def download_drive_file(service, file_info: dict) -> bytes:
    """Drive dosyasını indirir. Google Docs ise text/plain olarak export eder."""
    mime = file_info.get('mimeType', '')
    if mime == 'application/vnd.google-apps.document':
        # Google Docs → export
        req = service.files().export_media(fileId=file_info['id'], mimeType='text/plain')
    else:
        req = service.files().get_media(fileId=file_info['id'])
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

def extract_text_from_drive_file(data: bytes, mime: str) -> str:
    """Dosya verisinden metin çıkarır."""
    ext = DRIVE_MIME_MAP.get(mime, '')
    if ext == 'pdf':
        return extract_pdf(data)
    elif ext == 'docx':
        return extract_docx_b(data)
    elif ext == 'epub':
        return extract_epub_b(data)
    elif ext in ('txt', 'md', 'gdoc') or mime.startswith('text/'):
        return data.decode('utf-8', errors='ignore')
    return ""

def get_indexed_drive_ids() -> set[str]:
    """Supabase'deki Drive kaynaklı dosya ID'lerini döner."""
    try:
        res = supabase.table("documents").select("metadata").eq("metadata->>type", "drive").execute()
        return {d["metadata"]["drive_id"] for d in res.data if d.get("metadata", {}).get("drive_id")}
    except Exception as e:
        logging.error(f"Error getting indexed drive ids: {e}")
        return set()

def remove_drive_file_from_index(drive_id: str):
    """Belirli bir Drive dosyasının tüm chunk'larını Supabase'den siler."""
    try:
        supabase.table("documents").delete().eq("metadata->>drive_id", drive_id).execute()
    except Exception as e:
        logging.error(f"Error removing drive file from index: {e}")

async def sync_drive_folder(api_key: str, drive_folder_id: Optional[str] = None):
    """Drive klasörünü senkronize eder → yeni/değişen dosyaları indexler, silinenleri kaldırır."""
    global drive_sync_status
    
    folder_id_to_sync = drive_folder_id or GOOGLE_DRIVE_FOLDER_ID
    if not folder_id_to_sync:
        drive_sync_status.update({
            "status": "error",
            "message": "Drive klasör ID bulunamadı",
        })
        return

    drive_sync_status.update({
        "status": "running",
        "message": "Drive klasörü taranıyor...",
        "files_synced": 0,
        "total_chunks": 0,
        "files": [],
    })

    try:
        service = await asyncio.to_thread(get_drive_service)
        
        # Eğer kullanıcı klasör ismi girdiyse ID'yi çözümle
        try:
            resolved_folder_id = await asyncio.to_thread(resolve_drive_folder_id, service, folder_id_to_sync)
        except ValueError as e:
            drive_sync_status.update({
                "status": "error",
                "message": str(e)
            })
            return
            
        drive_files = await asyncio.to_thread(list_drive_files, service, resolved_folder_id)

        # Mevcut index'teki drive dosya ID'leri
        indexed_ids = await asyncio.to_thread(get_indexed_drive_ids)
        drive_file_ids = {f['id'] for f in drive_files}

        # Silinen dosyaları kaldır
        removed_ids = indexed_ids - drive_file_ids
        for rid in removed_ids:
            await asyncio.to_thread(remove_drive_file_from_index, rid)

        # Mevcut index'teki dosya modifiedTime'larını kontrol et
        indexed_times = {}
        items_res_data = []
        try:
            res = await asyncio.to_thread(lambda: supabase.table("documents").select("metadata").eq("metadata->>type", "drive").execute())
            items_res_data = res.data
            for d in items_res_data:
                m = d.get("metadata", {})
                if m.get("drive_id"):
                    indexed_times[m["drive_id"]] = m.get("modified_time", "")
        except Exception as e:
            logging.error(f"Drive index items fetch error: {e}")

        synced = 0
        total_chunks = 0
        file_list = []

        for f in drive_files:
            fid = f['id']
            fname = f['name']
            mime = f.get('mimeType', '')
            mod_time = f.get('modifiedTime', '')

            # Zaten indexlenmiş ve değişmemiş → atla
            if fid in indexed_times and indexed_times[fid] == mod_time:
                # Mevcut chunk sayısını hesapla
                existing_chunks = sum(
                    1 for d in items_res_data if d.get("metadata", {}).get("drive_id") == fid
                )
                file_list.append({"name": fname, "chunks": existing_chunks, "drive_id": fid})
                total_chunks += existing_chunks
                continue

            # Değişmiş dosya → eski chunk'ları sil
            if fid in indexed_ids:
                await asyncio.to_thread(remove_drive_file_from_index, fid)

            drive_sync_status["message"] = f"İndiriliyor: {fname}..."

            try:
                data = await asyncio.to_thread(download_drive_file, service, f)
                text = extract_text_from_drive_file(data, mime)
                if not text.strip():
                    continue

                chunks = chunk_text(text)
                n = await index_chunks(
                    chunks, f"📁 {fname}",
                    {"type": "drive", "drive_id": fid, "modified_time": mod_time},
                    api_key
                )
                synced += 1
                total_chunks += n
                file_list.append({"name": fname, "chunks": n, "drive_id": fid})
                drive_sync_status.update({
                    "files_synced": synced,
                    "total_chunks": total_chunks,
                    "message": f"Indexlendi: {fname} ({n} parça)",
                })
            except Exception as e:
                logging.warning(f"Drive dosyası işlenemedi: {fname} — {e}")
                continue

        if file_list:
            try:
                folders = await asyncio.to_thread(load_folders)
                save_needed = False
                for fold in folders:
                    if fold["id"] == "f_default":
                        for fn in file_list:
                            doc_name = f"📁 {fn['name']}"
                            if doc_name not in fold.get("docs", []):
                                fold.setdefault("docs", []).append(doc_name)
                                save_needed = True
                        break
                if save_needed:
                    await asyncio.to_thread(save_folders, folders)
            except Exception as e:
                logging.warning(f"Klasör güncellenirken hata: {e}")

        drive_sync_status.update({
            "status": "done",
            "last_sync": datetime.now().isoformat(),
            "files_synced": synced,
            "total_chunks": total_chunks,
            "files": file_list,
            "message": f"✅ {synced} yeni dosya indexlendi, toplam {len(file_list)} dosya · {total_chunks} parça",
        })
    except Exception as e:
        drive_sync_status.update({
            "status": "error",
            "message": f"❌ Sync hatası: {e}",
        })
        logging.error(f"Drive sync error: {e}")


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())

# ─── YouTube ─────────────────────────────────────────────────────────────────

def yt_video_id(url: str) -> Optional[str]:
    for p in [r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",
              r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
              r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
              r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"]:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def fetch_yt(vid: str) -> tuple[str, str]:
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(vid)
        
        # Türkçe veya İngilizce altyazı bulmaya çalış
        try:
            transcript = transcript_list.find_transcript(['tr', 'en'])
        except Exception:
            # Bulamazsa ilk dildeki altyazıyı al
            transcript = list(transcript_list)[0]
        
        data = transcript.fetch()
        text = " ".join([snippet.text for snippet in data])
        text = re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", text)).strip()
        
    except Exception as e:
        raise ValueError(f"Altyazı bulunamadı veya kapalı. (Sistem Hatası: {str(e)[:80]})") from e

    try:
        r = requests.get(f"https://www.youtube.com/watch?v={vid}",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        m = re.search(r'"title":"([^"]+)"', r.text)
        title = m.group(1) if m else f"YouTube:{vid}"
    except Exception:
        title = f"YouTube:{vid}"
    return text, title

# ─── Layered BFS Crawler ─────────────────────────────────────────────────────

SKIP_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|pdf|zip|mp4|mp3|css|js|svg|ico|woff|woff2|ttf|exe|dmg)$", re.I)

def get_sitemap_urls(base: str) -> list[str]:
    parsed = urlparse(base)
    root   = f"{parsed.scheme}://{parsed.netloc}"
    urls   = []
    for path in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/"]:
        try:
            r = requests.get(root + path, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and "xml" in r.headers.get("content-type", ""):
                soup = BeautifulSoup(r.text, "xml")
                for stag in soup.find_all("sitemap"):
                    loc = stag.find("loc")
                    if loc:
                        try:
                            r2 = requests.get(loc.text.strip(), timeout=10,
                                              headers={"User-Agent": "Mozilla/5.0"})
                            s2 = BeautifulSoup(r2.text, "xml")
                            urls += [t.text.strip() for t in s2.find_all("loc")]
                        except Exception: pass
                urls += [t.text.strip() for t in soup.find_all("loc")
                         if not t.find_parent("sitemap")]
                if urls: break
        except Exception: continue
    domain = parsed.netloc
    return [u for u in urls if urlparse(u).netloc == domain]


def crawl_one_layer(
    frontier: list[str],       # Bu katmanın taranacak URL listesi
    visited:  set[str],
    domain:   str,
    session:  requests.Session,
    max_pages: int = PAGES_PER_LAYER,
) -> tuple[list[tuple[str,str]], list[str]]:
    """
    frontier listesinden max_pages kadar sayfa tara.
    Returns:
      pages      = [(url, text), ...]   — bu katmanda bulunan içerikler
      next_layer = [url, ...]           — sonraki katman için keşfedilen linkler
    """
    pages:      list[tuple[str,str]] = []
    next_layer: list[str]            = []
    seen_next:  set[str]             = set()

    for url in frontier:
        if len(pages) >= max_pages:
            break
        url = url.split("#")[0].rstrip("/")
        if url in visited or SKIP_EXT.search(url):
            continue
        visited.add(url)

        try:
            r  = session.get(url, timeout=12)
            ct = r.headers.get("content-type", "")
            if "html" not in ct:
                continue
            text = clean_html(r.text)
            if len(text.strip()) < 100:
                continue
            pages.append((url, text))

            # Sonraki katman için linkleri topla
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full = urljoin(url, href).split("#")[0].rstrip("/")
                if (urlparse(full).netloc == domain
                        and full not in visited
                        and full not in seen_next
                        and not SKIP_EXT.search(full)):
                    next_layer.append(full)
                    seen_next.add(full)

            time.sleep(CRAWL_DELAY_SEC)
        except Exception:
            continue

    return pages, next_layer


async def layered_crawl_and_index(
    start_url:  str,
    job_id:     str,
    api_key:    str,
    query_hint: str = "",
):
    """
    Katman katman tara + her katman sonrası otomatik skor kontrolü.
    crawl_jobs[job_id] canlı güncellenir.
    Kullanıcı 'deeper' isteği de buraya yansır (job["force_next"]).
    """
    domain  = urlparse(start_url).netloc
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"})

    visited:      set[str]  = set()
    total_pages:  int       = 0
    total_chunks: int       = 0
    layer_num:    int       = 0

    # Başlangıç frontier: sitemap varsa oradan, yoksa ana sayfa
    sitemap_urls = get_sitemap_urls(start_url)
    if sitemap_urls:
        frontier    = sitemap_urls
        using_sitemap = True
    else:
        frontier    = [start_url]
        using_sitemap = False

    crawl_jobs[job_id].update({
        "status": "running", "layer": 0,
        "total_pages": 0, "total_chunks": 0,
        "score": None, "auto_stopped": False,
        "force_next": False, "paused": False,
        "using_sitemap": using_sitemap,
        "domain": domain,
    })

    while frontier:
        layer_num += 1
        crawl_jobs[job_id].update({
            "layer": layer_num,
            "layer_status": f"Katman {layer_num} taranıyor ({min(len(frontier), PAGES_PER_LAYER)} sayfa)...",
            "paused": False,
        })

        # ── Tarama (sync → thread) ──────────────────────────────────────────
        pages, next_frontier = await asyncio.to_thread(
            crawl_one_layer, frontier[:PAGES_PER_LAYER * 3],
            visited, domain, session, PAGES_PER_LAYER)

        # ── Index ───────────────────────────────────────────────────────────
        crawl_jobs[job_id]["layer_status"] = f"Katman {layer_num} indexleniyor ({len(pages)} sayfa)..."
        for url, text in pages:
            chunks = chunk_text(text)
            n = await index_chunks(chunks, f"🌐 {domain}", {"type": "web", "url": url, "layer": layer_num}, api_key)
            total_chunks += n
        total_pages += len(pages)

        crawl_jobs[job_id].update({
            "total_pages":  total_pages,
            "total_chunks": total_chunks,
        })

        # ── Skor kontrolü ───────────────────────────────────────────────────
        score = await content_score(query_hint, api_key) if query_hint else None
        crawl_jobs[job_id]["score"] = round(score, 3) if score is not None else None

        # Sonraki katmana link kalmadıysa bitti
        if not next_frontier:
            crawl_jobs[job_id].update({
                "status": "done",
                "message": f"✅ Site tamamen tarandı — {total_pages} sayfa · {total_chunks} parça · {layer_num} katman",
                "auto_stopped": False,
            })
            return

        # Sitemap modunda tüm URL'ler tek frontier'da, katman kavramı yok → tek turda bit
        if using_sitemap:
            frontier = []  # sitemap zaten hepsini verdi
        else:
            frontier = next_frontier

        # ── Otomatik dur kararı ─────────────────────────────────────────────
        if query_hint and score is not None and score < AUTO_SCORE_THRESHOLD:
            crawl_jobs[job_id].update({
                "status": "paused",
                "paused": True,
                "auto_stopped": True,
                "layer_status": f"✅ Katman {layer_num} yeterli — skor {score:.2f}",
                "message": (f"Katman {layer_num} tamamlandı. "
                            f"Yeterli içerik bulundu (skor: {score:.2f}). "
                            f"Daha derine inmek ister misin?"),
            })
            # Kullanıcının 'deeper' veya 'stop' demesini bekle
            while True:
                await asyncio.sleep(1)
                job = crawl_jobs.get(job_id, {})
                if job.get("force_next"):
                    crawl_jobs[job_id]["force_next"] = False
                    break  # devam et
                if job.get("force_stop"):
                    crawl_jobs[job_id].update({"status": "done",
                        "message": f"✅ Durduruldu — {total_pages} sayfa · {total_chunks} parça"})
                    return
        else:
            # Otomatik dur değil → kısa "katman bitti" mesajı ver ve devam et
            score_txt = f" · skor {score:.2f}" if score is not None else ""
            crawl_jobs[job_id]["layer_status"] = (
                f"Katman {layer_num} bitti ({total_pages} sayfa{score_txt}), devam ediliyor...")
            await asyncio.sleep(0.2)  # UI'nın görmesi için mini bekleme

    crawl_jobs[job_id].update({
        "status": "done",
        "message": f"✅ {total_pages} sayfa · {total_chunks} parça · {layer_num} katman",
    })


# ─── Startup: Auto-sync Drive ────────────────────────────────────────────────

@app.on_event("startup")
async def startup_sync():
    """Uygulama başlatılınca Drive klasörünü otomatik sync et."""
    if GOOGLE_DRIVE_FOLDER_ID and GOOGLE_SERVICE_ACCOUNT_JSON:
        api_key = GEMINI_API_KEY_ENV
        if api_key:
            logging.info("🔄 Startup: Google Drive sync başlatılıyor...")
            asyncio.create_task(sync_drive_folder(api_key))
        else:
            logging.warning("Drive sync atlanıyor: GEMINI_API_KEY ayarlanmamış")
    else:
        logging.info("Drive sync atlanıyor: GOOGLE_DRIVE_FOLDER_ID veya credentials yok")

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return Path("static/index.html").read_text(encoding="utf-8")

@app.get("/debug-db")
async def debug_db():
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_key_start": SUPABASE_KEY[:15] if SUPABASE_KEY else None,
        "python_version": sys.version
    }

# ── File upload ──────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    folder_id: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None)
):
    api_key = get_api_key(x_api_key)
    if not api_key: raise HTTPException(400, "GEMINI_API_KEY ayarlanmamış")

    content  = await file.read()
    filename = file.filename or "unknown"
    size_mb  = len(content) / (1024 * 1024)

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413,
            f"⚠️ Dosya çok büyük: {size_mb:.1f} MB (limit: {MAX_FILE_SIZE_MB} MB).")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if   ext == "pdf":                  text = extract_pdf(content)
        elif ext == "docx":                 text = extract_docx_b(content)
        elif ext == "epub":                 text = extract_epub_b(content)
        elif ext in ("txt","md","markdown"):text = content.decode("utf-8", errors="ignore")
        else: raise HTTPException(400, f"Desteklenmeyen format: .{ext}")
    except HTTPException: raise
    except Exception as e: raise HTTPException(400, f"Dosya okunamadı: {e}")

    if not text.strip():
        raise HTTPException(400, "Dosyadan metin çıkarılamadı.")

    chunks = chunk_text(text)
    
    try:
        await index_chunks(chunks, filename, {"type": "file", "size_mb": round(size_mb, 2)}, api_key)
        
        # Klasöre ekleme mantığı
        if folder_id and folder_id != "f_default":
            folders = load_folders()
            for f in folders:
                if f["id"] == folder_id:
                    if filename not in f.get("docs", []):
                        f.setdefault("docs", []).append(filename)
                        save_folders(folders)
                    break
        return {"message": f"✅ '{filename}' yüklendi ({len(chunks)} parça · {size_mb:.1f} MB)", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(400, f"Yükleme tamamlanamadı: {e}")

# ── Single URL ───────────────────────────────────────────────────────────────

@app.post("/add-url")
async def add_url(
    url: str = Form(...),
    folder_id: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None)
):
    api_key = get_api_key(x_api_key)
    if not api_key: raise HTTPException(400, "GEMINI_API_KEY ayarlanmamış")

    vid = yt_video_id(url)
    if vid:
        try:
            text, title = await asyncio.to_thread(fetch_yt, vid)
            source, extra = f"▶ {title}", {"type": "youtube", "url": url}
        except Exception as e:
            msg = str(e)
            if "Could not retrieve a transcript" in msg or "Subtitles are disabled" in msg:
                raise HTTPException(400, "Vercel Sunucu Engeli: YouTube, Vercel IP'lerinden gelen isteklere (bot koruması nedeniyle) engel koyuyor. Bu özellik sadece lokalde çalışır.")
            raise HTTPException(400, f"YouTube transkripti alınamadı: {msg}")
    else:
        try:
            r      = await asyncio.to_thread(lambda: requests.get(url, timeout=15,
                        headers={"User-Agent": "Mozilla/5.0"}))
            text   = clean_html(r.text)
            source, extra = url, {"type": "web", "url": url}
        except Exception as e:
            raise HTTPException(400, f"URL okunamadı: {e}")

    if not text.strip(): raise HTTPException(400, "İçerik boş")
    chunks = chunk_text(text)
    await index_chunks(chunks, source, extra, api_key)
    
    # Klasör işlemleri
    target_fid = folder_id
    
    # YouTube videoları için varsayılan klasör logic
    if vid and (not target_fid or target_fid == "f_default"):
        folders = load_folders()
        yt_folder = next((f for f in folders if f["name"].lower() == "youtube"), None)
        if not yt_folder:
            yt_folder = {"id": f"f_{str(uuid.uuid4())[:8]}", "name": "YouTube", "docs": []}
            folders.append(yt_folder)
        target_fid = yt_folder["id"]
        save_folders(folders)
        
    # Standard Web URL'leri (Site) için varsayılan klasör logic
    elif not vid and (not target_fid or target_fid == "f_default"):
        folders = load_folders()
        web_folder = next((f for f in folders if f["name"].lower() == "web"), None)
        if not web_folder:
            web_folder = {"id": f"f_{str(uuid.uuid4())[:8]}", "name": "Web", "docs": []}
            folders.append(web_folder)
        target_fid = web_folder["id"]
        save_folders(folders)

    if target_fid and target_fid != "f_default":
        folders = load_folders()
        for f in folders:
            if f["id"] == target_fid:
                if source not in f.get("docs", []):
                    f.setdefault("docs", []).append(source)
                    save_folders(folders)
                break

    return {"message": f"✅ {'▶' if vid else '🌐'} Eklendi ({len(chunks)} parça)"}

# ── Layered site crawl ───────────────────────────────────────────────────────

@app.post("/crawl-site")
async def start_crawl(
    url:         str  = Form(...),
    query_hint:  str  = Form(""),          # Opsiyonel: ne arıyorum?
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_api_key: Optional[str] = Header(None)
):
    api_key = get_api_key(x_api_key)
    if not api_key: raise HTTPException(400, "GEMINI_API_KEY ayarlanmamış")
    p = urlparse(url)
    if not p.scheme or not p.netloc: raise HTTPException(400, "Geçersiz URL")

    job_id = str(uuid.uuid4())[:8]
    crawl_jobs[job_id] = {"status": "queued", "url": url, "query_hint": query_hint}
    background_tasks.add_task(layered_crawl_and_index, url, job_id, api_key, query_hint)
    return {"job_id": job_id, "message": "🕷️ Katmanlı tarama başladı"}

@app.get("/crawl-status/{job_id}")
async def crawl_status(job_id: str):
    if job_id not in crawl_jobs: raise HTTPException(404, "Job bulunamadı")
    return crawl_jobs[job_id]

@app.post("/crawl-deeper/{job_id}")
async def crawl_deeper(job_id: str):
    """Kullanıcı 'daha derine in' dedi."""
    if job_id not in crawl_jobs: raise HTTPException(404, "Job bulunamadı")
    crawl_jobs[job_id]["force_next"] = True
    crawl_jobs[job_id]["paused"]     = False
    return {"message": "Bir sonraki katmana geçiliyor..."}

@app.post("/crawl-stop/{job_id}")
async def crawl_stop(job_id: str):
    """Kullanıcı 'yeter' dedi."""
    if job_id not in crawl_jobs: raise HTTPException(404, "Job bulunamadı")
    crawl_jobs[job_id]["force_stop"] = True
    return {"message": "Tarama durduruluyor..."}

HISTORY_FILE = "history.json"

def load_history() -> list:
    try:
        res = supabase.table("folders").select("docs").eq("id", "f_history").limit(1).execute()
        if res.data and isinstance(res.data[0].get("docs"), list):
            return res.data[0]["docs"]
    except Exception as e:
        logging.error(f"Error loading history from Supabase: {e}")
    return []

def save_history(history: list):
    try:
        # Check if f_history exists
        res = supabase.table("folders").select("id").eq("id", "f_history").limit(1).execute()
        if not res.data:
            supabase.table("folders").insert({
                "id": "f_history",
                "name": "_System_History_",
                "docs": history
            }).execute()
        else:
            supabase.table("folders").update({"docs": history}).eq("id", "f_history").execute()
    except Exception as e:
        logging.error(f"Error saving history to Supabase: {e}")

# ── Query ────────────────────────────────────────────────────────────────────

@app.post("/query")
async def query(
    question:         str = Form(...),
    system_prompt:    str = Form(DEFAULT_SYSTEM_PROMPT),
    top_k:            int = Form(5),
    selected_sources: str = Form(""),   # virgülle ayrılmış kaynak isimleri (boş=hepsi)
    x_api_key: Optional[str] = Header(None)
):
    api_key = get_api_key(x_api_key)
    if not api_key: raise HTTPException(400, "GEMINI_API_KEY ayarlanmamış")
    
    # Supabase'de belge var mı?
    count_res = await asyncio.to_thread(lambda: supabase.table("documents").select("id", count="exact").limit(1).execute())
    if not count_res.count or count_res.count == 0: 
        raise HTTPException(400, "Henüz belge yüklenmedi")

    try:
        q_emb = await get_embedding(question, api_key)
    except Exception as e:
        raise HTTPException(400, f"Embedding hatası (API Key geçersiz olabilir): {e}")

    q_emb_str = f"[{','.join(map(str, q_emb))}]"
    
    # Vector search RPC call
    try:
        res = await asyncio.to_thread(
            lambda: supabase.rpc('match_documents', {
                'query_embedding': q_emb_str,
                'match_threshold': 0.0,
                'match_count': top_k
            }).execute()
        )
    except Exception as e:
        logging.error(f"Supabase match_documents RPC error: {e}")
        raise HTTPException(500, f"Veritabanı arama hatası (Fonksiyon yüklendi mi?): {e}")

    results_data = res.data or []
    
    # Filtreleme (where_filter)
    if selected_sources.strip():
        src_list = [s.strip() for s in selected_sources.split("||||") if s.strip()]
        if src_list:
            results_data = [d for d in results_data if d.get("source") in src_list]

    if not results_data:
        # Sonuç bulunamasa da LLM'ye boş veya en iyi tahmini yollayalım
        chunks = ["(Belirli kriterlere uyan veri bulunamadı)"]
        metas = []
        dists = []
        sources = []
    else:
        chunks  = [d["content"] for d in results_data]
        metas   = [{"source": d["source"]} for d in results_data]
        # cosine distance = 1 - similarity
        dists   = [1.0 - d["similarity"] for d in results_data]
        sources = list({d["source"] for d in results_data})

    # Chunk detayları (popup için)
    chunk_details = []
    for i, c in enumerate(chunks):
        chunk_details.append({
            "id": i + 1,
            "text": c,
            "source": metas[i]["source"],
            "score": round(dists[i], 3) if i < len(dists) else None,
        })

    context = "\n\n---\n\n".join(
        f"[Kaynak {i+1}]: {c}" for i, c in enumerate(chunks))
    try:
        answer  = await gemini_chat(
            f"Aşağıdaki kaynaklardan yararlanarak soruyu yanıtla:\n\n{context}\n\nSoru: {question}",
            system_prompt, api_key)
    except Exception as e:
        raise HTTPException(500, f"Yapay zeka yanıt üretirken hata oluştu: {e}")

    # Markdown temizle
    answer = strip_markdown(answer)

    # Geçmişe kaydet
    history_item = {
        "id": str(uuid.uuid4())[:8],
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "chunk_details": chunk_details
    }
    hist = load_history()
    hist.insert(0, history_item) # En başa ekle
    save_history(hist[:50]) # Son 50 soruyu tut

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "chunk_details": chunk_details,
    }

@app.get("/history")
async def get_history():
    return load_history()

@app.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    hist = load_history()
    new_hist = [h for h in hist if h.get("id") != item_id]
    if len(hist) == len(new_hist):
        raise HTTPException(404, "Geçmiş kaydı bulunamadı")
    save_history(new_hist)
    return {"message": "Geçmiş kaydı silindi"}

@app.post("/history/{item_id}/touch")
async def touch_history_item(item_id: str):
    hist = load_history()
    for i, h in enumerate(hist):
        if h.get("id") == item_id:
            item = hist.pop(i)
            hist.insert(0, item)
            save_history(hist)
            return {"message": "Geçmiş öne alındı"}
    raise HTTPException(404, "Geçmiş kaydı bulunamadı")

# ── Documents & Folders ──────────────────────────────────────────────────────

def load_folders() -> list[dict]:
    """Supabase'den klasör listesini çeker."""
    try:
        res = supabase.table("folders").select("*").execute()
        folders = []
        has_default = False
        for row in (res.data or []):
            if row["id"] == "f_history": continue
            docs = row.get("docs") or []
            if isinstance(docs, str):
                try: docs = json.loads(docs)
                except: docs = []
            f = {"id": row["id"], "name": row["name"], "docs": docs}
            if row["id"] == "f_default":
                has_default = True
            folders.append(f)
        if not has_default:
            default_row = {"id": "f_default", "name": "Diğer Belgeler", "docs": []}
            try:
                supabase.table("folders").upsert(default_row).execute()
            except: pass
            folders.insert(0, default_row)
        return folders
    except Exception as e:
        logging.error(f"load_folders error: {e}")
        return [{"id": "f_default", "name": "Diğer Belgeler", "docs": []}]

def save_folders(data: list[dict]):
    """Klasör verilerini Supabase'e yazar (upsert)."""
    try:
        for f in data:
            if f["id"] == "f_history": continue
            row = {"id": f["id"], "name": f["name"], "docs": f.get("docs", [])}
            supabase.table("folders").upsert(row).execute()
    except Exception as e:
        logging.error(f"save_folders error: {e}")

@app.get("/folders")
async def get_folders():
    folders = load_folders()
    
    # Supabase'den belge kaynaklarını getir (select distinct source gibi)
    # Ancak Supabase REST'te distinct zor. Şimdilik düz select yapıp python'da gruplayalım
    docs_res = await asyncio.to_thread(lambda: supabase.table("documents").select("source, metadata").execute())
    
    if not docs_res.data:
        return {"folders": folders, "total_chunks": 0}
        
    srcs: dict[str, dict] = {}
    for d in docs_res.data:
        s = d["source"]
        if s not in srcs:
            m = d.get("metadata", {})
            srcs[s] = {"name": s, "chunks": 0, "type": m.get("type","file")}
        srcs[s]["chunks"] += 1
    
    all_docs = list(srcs.values())
    
    # Hangi belge hangi klasörde bulalım
    assigned_docs = set()
    for f in folders:
        # Sadece yüklenmiş belgeleri klasörde göster
        f["doc_details"] = [doc for doc in all_docs if doc["name"] in f.get("docs", [])]
        assigned_docs.update(f.get("docs", []))
    
    # Hiçbir klasöre atanmamış belgeleri "Diğer Belgeler" (f_default) içine ekle
    unassigned = [doc for doc in all_docs if doc["name"] not in assigned_docs]
    for f in folders:
        if f["id"] == "f_default":
            f["doc_details"].extend(unassigned)
            break

    # Supabase count
    count_res = await asyncio.to_thread(lambda: supabase.table("documents").select("id", count="exact").limit(1).execute())
    total_chunks = count_res.count if count_res.count else 0

    return {"folders": folders, "total_chunks": total_chunks}

@app.post("/create-folder")
async def create_folder(name: str = Form(...)):
    name = name.strip()
    if not name: raise HTTPException(400, "Klasör ismi boş olamaz")
    folders = load_folders()
    folders.append({
        "id": f"f_{str(uuid.uuid4())[:8]}",
        "name": name,
        "docs": []
    })
    save_folders(folders)
    return {"message": f"📁 '{name}' oluşturuldu"}

@app.post("/rename-folder")
async def rename_folder(folder_id: str = Form(...), new_name: str = Form(...)):
    if folder_id == "f_default": raise HTTPException(400, "Varsayılan klasör adı değiştirilemez")
    new_name = new_name.strip()
    if not new_name: raise HTTPException(400, "Yeni isim boş olamaz")
    folders = load_folders()
    for f in folders:
        if f["id"] == folder_id:
            f["name"] = new_name
            save_folders(folders)
            return {"message": "✅ Klasör adı güncellendi"}
    raise HTTPException(404, "Klasör bulunamadı")

@app.delete("/folder/{folder_id}")
async def delete_folder(folder_id: str):
    if folder_id == "f_default": raise HTTPException(400, "Varsayılan klasör silinemez")
    folders = load_folders()
    new_folders = [f for f in folders if f["id"] != folder_id]
    if len(folders) == len(new_folders):
        raise HTTPException(404, "Klasör bulunamadı")
    save_folders(new_folders)
    return {"message": "🗑️ Klasör silindi. İçindeki belgeler 'Diğer Belgeler'e aktarıldı."}

@app.post("/move-doc")
async def move_doc(doc_name: str = Form(...), target_folder_id: str = Form(...)):
    folders = load_folders()
    # Önce belgeyi bulunduğu tüm klasörlerden çıkar
    for f in folders:
        if doc_name in f.get("docs", []):
            f["docs"].remove(doc_name)
    
    # Şimdi hedefe ekle (f_default ise hiçbir yere eklemeyiz, dinamik olarak oraya düşer)
    if target_folder_id != "f_default":
        for f in folders:
            if f["id"] == target_folder_id:
                if "docs" not in f: f["docs"] = []
                if doc_name not in f["docs"]:
                    f["docs"].append(doc_name)
                break
    
    save_folders(folders)
    return {"message": f"✅ Belge taşındı"}

# ── Documents (Eski uç nokta - geriye dönük uyumluluk için korundu) ─────────

@app.get("/documents")
async def list_documents():
    docs_res = await asyncio.to_thread(lambda: supabase.table("documents").select("source, metadata").execute())
    if not docs_res.data:
        return {"documents": [], "total_chunks": 0}
    srcs: dict[str, dict] = {}
    for d in docs_res.data:
        s = d["source"]
        if s not in srcs:
            m = d.get("metadata", {})
            srcs[s] = {"name": s, "chunks": 0, "type": m.get("type","file")}
        srcs[s]["chunks"] += 1
    count_res = await asyncio.to_thread(lambda: supabase.table("documents").select("id", count="exact").limit(1).execute())
    total = count_res.count if count_res.count else 0
    return {"documents": list(srcs.values()), "total_chunks": total}

@app.post("/rename-document")
async def rename_document(
    old_name: str = Form(...),
    new_name: str = Form(...),
):
    """Belge kaynağının adını değiştirir (Supabase)."""
    # Supabase'deki source alanını güncelle
    res = await asyncio.to_thread(lambda: supabase.table("documents").update({"source": new_name}).eq("source", old_name).execute())
    if not res.data:
        raise HTTPException(404, f"'{old_name}' bulunamadı")
    
    # Klasör referanslarını güncelle
    folders = load_folders()
    for f in folders:
        if old_name in f.get("docs", []):
            f["docs"].remove(old_name)
            f["docs"].append(new_name)
    save_folders(folders)
    
    return {"message": f"✅ '{old_name}' → '{new_name}' olarak güncellendi ({len(res.data)} parça)"}

@app.delete("/document/{doc_name:path}")
async def delete_document(doc_name: str):
    import urllib.parse
    doc_name = urllib.parse.unquote(doc_name)

    # Supabase Delete
    res = await asyncio.to_thread(lambda: supabase.table("documents").delete().eq("source", doc_name).execute())
    if not res.data:
        raise HTTPException(404, f"'{doc_name}' bulunamadı")
    
    folders = load_folders()
    for f in folders:
        if doc_name in f.get("docs", []):
            f["docs"].remove(doc_name)
    save_folders(folders)
    
    return {"message": f"🗑️ '{doc_name}' tamamen silindi ({len(res.data)} parça)"}

@app.delete("/documents")
async def clear_all():
    # Supabase clear all
    # Truncate veya hepsini silme (büyükse limit gelir, şimdilik dummy)
    await asyncio.to_thread(lambda: supabase.table("documents").delete().neq("id", "0").execute())
    
    # Klasörleri de sıfırla
    save_folders([{"id": "f_default", "name": "Diğer Belgeler", "docs": []}])
    
    return {"message": "Tüm belgeler silindi"}

# ── Document Viewer & Summarization ──────────────────────────────────────────

@app.get("/document/{doc_name:path}")
async def get_document_content(doc_name: str):
    import urllib.parse
    doc_name = urllib.parse.unquote(doc_name)
    
    res = await asyncio.to_thread(lambda: supabase.table("documents").select("content").eq("source", doc_name).execute())
    if not res.data:
        raise HTTPException(404, "Belge bulunamadı veya içeriği yok")
        
    full_text = "\n\n".join(d["content"] for d in res.data)
    return {"name": doc_name, "content": full_text}

@app.post("/summarize-document")
async def summarize_document(
    doc_name: str = Form(...),
    x_api_key: Optional[str] = Header(None)
):
    api_key = get_api_key(x_api_key)
    if not api_key:
        raise HTTPException(401, "API Key eksik")
    
    res = await asyncio.to_thread(lambda: supabase.table("documents").select("content").eq("source", doc_name).execute())
    if not res.data:
        raise HTTPException(404, "Belge bulunamadı")
        
    full_text = "\n\n".join(d["content"] for d in res.data)
    max_chars = 100000 
    truncated_text = full_text[:max_chars]
    
    prompt = f"Lütfen aşağıdaki belgenin ana hatlarıyla kapsamlı bir özetini çıkar. Belge ne hakkında ve hangi detayları içeriyor kısaca belirt:\n\n{truncated_text}"
    
    try:
        summary_text = await gemini_chat(prompt, "Sen bir akademik özetleme asistanısın. Belgeyi nesnel, açık ve anlaşılır bir şekilde özetle. Kısa maddeler kullanabilirsin.", api_key)
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(500, f"Özet çıkarılırken hata oluştu: {str(e)}")

@app.post("/toc-document")
async def toc_document(
    doc_name: str = Form(...),
    x_api_key: Optional[str] = Header(None)
):
    """Belgenin içindekiler listesini (TOC) çıkarır."""
    api_key = get_api_key(x_api_key)
    if not api_key:
        raise HTTPException(401, "API Key eksik")
    
    res = await asyncio.to_thread(lambda: supabase.table("documents").select("content").eq("source", doc_name).execute())
    if not res.data:
        raise HTTPException(404, "Belge bulunamadı")
        
    full_text = "\n\n".join(d["content"] for d in res.data)
    max_chars = 100000 
    truncated_text = full_text[:max_chars]
    
    prompt = f"Lütfen aşağıdaki belgenin sadece 'Ana Başlıkları' ve 'Alt Başlıklarını' hiyerarşik bir liste (İçindekiler / Table of Contents) formatında çıkar. Bütün başlıkları KESİNLİKLE TÜRKÇE DİLİNDE oluştur:\n\n{truncated_text}"
    
    try:
        toc_text = await gemini_chat(prompt, "Sen bir akademik asistansın. Sadece başlıkları kullanarak hiyerarşik bir İçindekiler listesi oluştur. Madde işareti kullan, ekstra yorum yapma.", api_key)
        return {"toc": toc_text}
    except Exception as e:
        raise HTTPException(500, f"İçindekiler çıkarılırken hata oluştu: {str(e)}")


# ── Google Drive Sync ────────────────────────────────────────────────────────

@app.post("/sync-drive")
async def sync_drive(
    background_tasks: BackgroundTasks,
    folder_id: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None)
):
    """Manuel Drive sync tetikler."""
    if not GOOGLE_DRIVE_FOLDER_ID and not folder_id:
        raise HTTPException(400, "GOOGLE_DRIVE_FOLDER_ID veya özel klasör ID ayarlanmamış")
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise HTTPException(400, "GOOGLE_SERVICE_ACCOUNT_JSON ayarlanmamış")
    if drive_sync_status["status"] == "running":
        raise HTTPException(409, "Sync zaten çalışıyor")

    api_key = get_api_key(x_api_key)
    if not api_key:
        raise HTTPException(400, "GEMINI_API_KEY ayarlanmamış")

    folder_id_to_sync = folder_id or GOOGLE_DRIVE_FOLDER_ID
    background_tasks.add_task(sync_drive_folder, api_key, folder_id_to_sync)
    return {"message": "🔄 Drive sync başlatıldı"}

@app.get("/drive-status")
async def get_drive_status():
    """Drive sync durumunu döner."""
    return {
        **drive_sync_status,
        "drive_configured": bool(GOOGLE_DRIVE_FOLDER_ID and GOOGLE_SERVICE_ACCOUNT_JSON),
    }

app.mount("/static", StaticFiles(directory="static"), name="static")
