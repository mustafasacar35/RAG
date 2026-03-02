import sys
print(sys.executable)
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    print(dir(YouTubeTranscriptApi))
    tl = YouTubeTranscriptApi.list_transcripts("bS5P_LAqiVg")
    for t in tl:
        print(t.language, t.language_code, t.is_generated)
except Exception as e:
    print("Error:", e)
