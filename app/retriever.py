import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import EMBED_MODEL, CROSS_ENCODER, TOP_K
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self):
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.faiss_index = faiss.read_index("data/indexes/faiss.index")
        with open("data/indexes/meta.pkl", "rb") as f:
            self.meta = pickle.load(f)
        with open("data/indexes/bm25.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_meta = bm25_data["metas"]
        self.cross_encoder = CrossEncoder(CROSS_ENCODER) if CROSS_ENCODER else None

    def retrieve(self, query, top_k=TOP_K):
        q_vec = self.embed_model.encode([query], convert_to_numpy=True)
        D, I = self.faiss_index.search(q_vec, top_k)
        faiss_results = [{"score": float(1 - d), **self.meta[i]} for i, d in zip(I[0], D[0])]

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_results = sorted(
            [{"score": float(s), **m} for s, m in zip(bm25_scores, self.bm25_meta)],
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        combined = {r["text"]: r for r in (faiss_results + bm25_results)}
        results = list(combined.values())

        if self.cross_encoder:
            pairs = [(query, r["text"]) for r in results]
            ce_scores = self.cross_encoder.predict(pairs)
            for r, s in zip(results, ce_scores):
                r["score"] = float(s)
            results.sort(key=lambda x: x["score"], reverse=True)
        else:
            results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
