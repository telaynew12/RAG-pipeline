import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import EMBED_MODEL, CROSS_ENCODER, TOP_K
from rank_bm25 import BM25Okapi
from some_llm_library import LLM  # Replace with your LLM, e.g., OpenAI, HuggingFace

class HybridRetriever:
    def __init__(self):
        # Embedding model for FAISS
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        
        # Load FAISS index and metadata
        self.faiss_index = faiss.read_index("data/indexes/faiss.index")
        with open("data/indexes/meta.pkl", "rb") as f:
            self.meta = pickle.load(f)
        
        # Load BM25 index and metadata
        with open("data/indexes/bm25.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_meta = bm25_data["metas"]
        
        # Optional cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder(CROSS_ENCODER) if CROSS_ENCODER else None
        
        # Initialize your LLM
        self.llm = LLM()  # Replace with actual LLM initialization

    def retrieve(self, query, top_k=TOP_K):
        # ------------------ FAISS Semantic Search ------------------
        q_vec = self.embed_model.encode([query], convert_to_numpy=True)
        D, I = self.faiss_index.search(q_vec, top_k)
        faiss_results = [{"score": float(1 - d), **self.meta[i]} for i, d in zip(I[0], D[0])]

        # ------------------ BM25 Keyword Search ------------------
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_results = sorted(
            [{"score": float(s), **m} for s, m in zip(bm25_scores, self.bm25_meta)],
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        # ------------------ Combine & Deduplicate ------------------
        combined = {r["text"]: r for r in (faiss_results + bm25_results)}
        results = list(combined.values())

        # ------------------ Optional Cross-Encoder Re-Ranking ------------------
        if self.cross_encoder:
            pairs = [(query, r["text"]) for r in results]
            ce_scores = self.cross_encoder.predict(pairs)
            for r, s in zip(results, ce_scores):
                r["score"] = float(s)
            results.sort(key=lambda x: x["score"], reverse=True)
        else:
            results.sort(key=lambda x: x["score"], reverse=True)

        top_chunks = results[:top_k]

        # ------------------ Prepare Context for LLM (metadata-for-reasoning-only) ------------------
        # Include Objective/KeyResult in parentheses for reasoning, but instruct LLM not to output them
        context = "\n".join([
            f"{r['text']} (Objective: {r.get('objective','')}, KeyResult: {r.get('keyresult','')})"
            for r in top_chunks
        ])

        prompt = f"""Using the context below, answer the question.
You may use the Objective and KeyResult references internally to decide relevance, but **do NOT include them in your final answer**.

Context:
{context}

Question:
{query}"""

        llm_response = self.llm.generate(prompt)  # Replace with your LLM's generate/predict method

        return {
            "query": query,
            "top_chunks": top_chunks,
            "llm_response": llm_response
        }
