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
    print(f"✅ Created {len(chunks)} chunks.")
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

# ----------------------
# Hybrid Retriever for Streamlit
# ----------------------
class HybridRetriever:
    def __init__(self):
        # Load FAISS
        faiss_index_path = INDEX_DIR / "faiss.index"
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        with open(INDEX_DIR / "meta.pkl", "rb") as f:
            self.faiss_metas = pickle.load(f)
        self.embedding_model = SentenceTransformer(EMBED_MODEL)

        # Load BM25
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_metas = bm25_data["metas"]

    def retrieve(self, query, top_k=5):
        # FAISS semantic search
        q_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(q_vector, top_k)
        faiss_results = [self.faiss_metas[i]["text"] for i in indices[0]]

        # BM25 keyword search
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        bm25_results = [self.bm25_metas[i]["text"] for i in top_indices]

        # Combine results and deduplicate
        combined = list(dict.fromkeys(faiss_results + bm25_results))
        return [{"text": t} for t in combined[:top_k]]


if __name__ == "__main__":
    chunks = ingest_and_chunk()
    build_faiss(chunks)
    build_bm25(chunks)
    print("🎯 All indexes built successfully.")
