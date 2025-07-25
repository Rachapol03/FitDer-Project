import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # โหลด .env

try:
    api_key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    print(f"API Key loaded: {api_key[:5]}...{api_key[-5:]}")

    print("\nAvailable Models and their supported methods:")
    for m in genai.list_models():
        # กรองเฉพาะโมเดลที่รองรับ generateContent()
        if "generateContent" in m.supported_generation_methods:
            print(f"Name: {m.name}, Methods: {m.supported_generation_methods}")

except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")