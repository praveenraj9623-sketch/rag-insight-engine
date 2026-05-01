import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("AIzaSyDb34xJOmW3X6IIdfBeRx2lOWEkhIRkgYk")

# Keep everything else SAME if already exists
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

VECTOR_DB_PATH = "vector_store/faiss_index"