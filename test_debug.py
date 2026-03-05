import asyncio
import sys
from google import genai
from google.genai import types

async def main():
    api_key = sys.argv[1] if len(sys.argv) > 1 else "AlzaSyCJDP4zKdtWLHjT3xnP3VSBnK-8XexAVuE"
    
    print(f"Testing with API Key: {api_key[:10]}...")
    client = genai.Client(api_key=api_key)
    
    print("Trying text-embedding-004...")
    try:
        r1 = await asyncio.to_thread(
            client.models.embed_content,
            model='text-embedding-004',
            contents="test",
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        print("Success!", len(r1.embeddings[0].values))
    except Exception as e:
        print("Error with text-embedding-004:", repr(e))
        
    print("\nTrying embedding-001...")
    try:
        r2 = await asyncio.to_thread(
            client.models.embed_content,
            model='models/embedding-001',
            contents="test"
        )
        print("Success!", len(r2.embeddings[0].values))
    except Exception as e:
        print("Error with embedding-001:", repr(e))

if __name__ == "__main__":
    asyncio.run(main())
