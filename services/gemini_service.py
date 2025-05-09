import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config = {
        "temperature": 0
    }
)

def generate_gemini_response(prompt: str, history: list = None) -> str:
    chat = model_gemini.start_chat(history=history or [])
    response = chat.send_message(prompt)
    return response.text.strip()
