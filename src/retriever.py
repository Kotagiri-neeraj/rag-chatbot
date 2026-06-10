from typing import List, Tuple
from src.vectorstore import load_vectorstore, load_documents

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


class SimpleBM25Retriever:
    def __init__(self, documents: List):
        # documents: list of Document objects
        self.documents = documents
        texts = [getattr(d, "page_content", "") or "" for d in documents]
        # simple whitespace tokenizer
        tokenized = [t.lower().split() for t in texts]
        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 is required for BM25 retriever. Install with `pip install rank-bm25`.")
        self.bm25 = BM25Okapi(tokenized)

    def get_relevant_documents(self, query: str, k: int = 5):
        qtokens = query.lower().split()
        scores = self.bm25.get_scores(qtokens)
        # get top k indices
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [self.documents[i] for i in top_idx]
        # attach score in metadata for later combination
        scored = []
        for i in top_idx:
            doc = self.documents[i]
            meta = getattr(doc, "metadata", {}) or {}
            meta = dict(meta)
            meta.setdefault("_bm25_score", float(scores[i]))
            # create a shallow wrapper Document-like object
            scored.append(doc)
        return scored


class HybridEnsembleRetriever:
    def __init__(self, bm25_retriever: SimpleBM25Retriever, faiss_db, k_bm25: int = 5, k_faiss: int = 5, weights=(0.5, 0.5)):
        self.bm25 = bm25_retriever
        self.db = faiss_db
        self.k_bm25 = k_bm25
        self.k_faiss = k_faiss
        self.weights = weights

    def get_relevant_documents(self, query: str) -> List:
        # BM25 candidates with scores
        bm25_docs = self.bm25.get_relevant_documents(query, k=self.k_bm25)

        # FAISS similarity search with scores if available
        faiss_pairs: List[Tuple] = []
        try:
            faiss_pairs = self.db.similarity_search_with_score(query, k=self.k_faiss)
        except Exception:
            try:
                # try retriever interface
                retr = self.db.as_retriever()
                faiss_docs = retr.get_relevant_documents(query)
                faiss_pairs = [(d, 0.0) for d in faiss_docs]
            except Exception:
                faiss_pairs = []

        # normalize and combine scores
        combined = {}

        # BM25 scores
        bm25_scores = []
        for d in bm25_docs:
            s = (d.metadata or {}).get("_bm25_score", 0.0)
            bm25_scores.append(s)
        max_b = max(bm25_scores) if bm25_scores else 1.0

        for d in bm25_docs:
            key = self._doc_key(d)
            s = (d.metadata or {}).get("_bm25_score", 0.0) / max_b if max_b else 0.0
            combined[key] = {"doc": d, "score": s * self.weights[0]}

        # FAISS scores
        faiss_scores = [s for _, s in faiss_pairs] if faiss_pairs else []
        max_f = max(faiss_scores) if faiss_scores else 1.0
        for d, raw in faiss_pairs:
            key = self._doc_key(d)
            s = (raw / max_f) if max_f else 0.0
            if key in combined:
                combined[key]["score"] += s * self.weights[1]
            else:
                combined[key] = {"doc": d, "score": s * self.weights[1]}

        # sort by combined score
        sorted_docs = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def _doc_key(self, doc):
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source")
        page = meta.get("page")
        if src or page:
            return f"{src}::{page}"
        return str(hash(getattr(doc, "page_content", "")))


def build_hybrid_retriever(k_bm25: int = 5, k_faiss: int = 5, weights=(0.5, 0.5)):
    """Compatibility hybrid retriever using rank_bm25 and FAISS directly.

    Returns an object exposing `get_relevant_documents(query)`.
    """
    documents = load_documents()
    bm25 = SimpleBM25Retriever(documents)
    db = load_vectorstore()
    return HybridEnsembleRetriever(bm25_retriever=bm25, faiss_db=db, k_bm25=k_bm25, k_faiss=k_faiss, weights=weights)
