import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

gemini_api_key2 = os.getenv("GEMINI_API_KEY_2")
if not gemini_api_key2:
    gemini_api_key2 = gemini_api_key

def get_llm():
    llm = ChatGoogleGenerativeAI(
        # model="gemini-2.5-flash-preview-04-17",
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key
    )
    return llm

def get_llm_2():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key2
    )
    return llm