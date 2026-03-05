import os
import sys
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # Use the key provided by the user in the prompt if available, or ask them to set it.
    api_key = sys.argv[1] if len(sys.argv) > 1 else None

if not api_key:
    print("Please provide an API key.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

try:
    print("Available embedding models:")
    for m in client.models.list():
        if 'embedContent' in m.supported_actions:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
