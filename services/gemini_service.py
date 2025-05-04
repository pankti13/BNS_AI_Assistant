import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash")

def generate_gemini_response(prompt: str) -> str:
    response = model_gemini.generate_content(prompt)
    return response.text.strip()
