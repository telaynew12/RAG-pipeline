import os
import pickle
from pathlib import Path
from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

from app.config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def ingest_and_chunk():
    print("📥 Loading documents...")
    loaders = [
        DirectoryLoader(str(RAW_DIR), glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(str(RAW_DIR), glob="*.pdf", loader_cls=PyPDFLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    print(f"✅ Loaded {len(docs)} documents. Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return chunks

def build_faiss(chunks):
    print("🔍 Building FAISS index...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c.page_content for c in chunks]
    metas = [{"source": c.metadata.get("source", ""), "text": c.page_content} for c in chunks]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "meta.pkl", "wb") as f:
        pickle.dump(metas, f)
    print("✅ FAISS index ready.")

def build_bm25(chunks):
    print("🔍 Building BM25 index...")
    texts = [c.page_content for c in chunks]
    metas = [{"source": c.metadata.get("source", ""), "text": c.page_content} for c in chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "metas": metas}, f)
    print("✅ BM25 index ready.")

if __name__ == "__main__":
    chunks = ingest_and_chunk()
    build_faiss(chunks)
    build_bm25(chunks)
    print("🎯 All indexes built successfully.")
