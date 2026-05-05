import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_gemini_api_key():
    """
    Loads Gemini API key safely.

    Local development:
        Uses .env file

    Streamlit Cloud:
        Uses Streamlit Secrets
    """

    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


GEMINI_API_KEY = get_gemini_api_key()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

VECTOR_DB_PATH = "vector_store/faiss_index"