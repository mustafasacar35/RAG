import asyncio
import os
import sys
from google import genai
from google.genai import types

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set")
        return
        
    client = genai.Client(api_key=api_key.strip())
    text = "hello world"
    
    print("--- OLD LOGIC ---")
    vals = None
    try:
        response = await asyncio.to_thread(
            client.models.embed_content,
            model='text-embedding-004',
            contents=text[:8000],
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        vals = response.embeddings[0].values
        print("OLD Logic text-embedding-004 Success! length:", len(vals))
    except Exception as e:
        print("OLD Logic text-embedding-004 failed:", e)
        try:
            response = await asyncio.to_thread(
                client.models.embed_content,
                model='models/embedding-001',
                contents=text[:8000]
            )
            vals = response.embeddings[0].values
            print("OLD Logic models/embedding-001 Success! length:", len(vals))
        except Exception as e2:
            print("OLD Logic models/embedding-001 failed:", e2)

    print("\n--- NEW LOGIC ---")
    models_to_try = ['text-embedding-004', 'models/text-embedding-004', 'embedding-001', 'models/embedding-001']
    for model_name in models_to_try:
        try:
            print(f"Trying {model_name}...")
            # Simulate the buggy config explicit passing
            response = await asyncio.to_thread(
                client.models.embed_content,
                model=model_name,
                contents=text[:8000],
                config=types.EmbedContentConfig(output_dimensionality=768) if '004' in model_name else None
            )
            vals = response.embeddings[0].values
            print(f"NEW Logic {model_name} Success! length: {len(vals)}")
            break
        except Exception as e:
            print(f"NEW Logic {model_name} failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
