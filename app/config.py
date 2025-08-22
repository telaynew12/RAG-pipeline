import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER = os.getenv("CROSS_ENCODER", None)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", None)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 10
