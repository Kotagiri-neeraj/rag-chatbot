from typing import List, Optional

class RerankingRetriever:
    """Wraps an existing retriever and reranks top candidates using a CrossEncoder.

    Usage: wrap an EnsembleRetriever (BM25+FAISS) so the final returned documents
    are the top-k after cross-encoder scoring.
    
    Falls back to no-reranking if CrossEncoder is unavailable.
    """
    def __init__(self, base_retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 6, final_k: int = 2, device: Optional[str] = None):
        self.base = base_retriever
        self.top_k = top_k
        self.final_k = final_k
        self.model = None
        self.enabled = False

        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # load cross-encoder on chosen device
            self.model = CrossEncoder(model_name, device=device)
            self.enabled = True
        except Exception as e:
            # Gracefully disable reranking if CrossEncoder is unavailable
            print(f"Warning: CrossEncoder reranking disabled ({str(e)[:60]}...). Using base retriever only.")

    def get_relevant_documents(self, query: str) -> List:
        # obtain candidates from base retriever
        if hasattr(self.base, "get_relevant_documents"):
            candidates = self.base.get_relevant_documents(query)
        elif callable(self.base):
            candidates = self.base(query)
        else:
            raise RuntimeError("Base retriever does not expose get_relevant_documents or is not callable.")

        if not candidates or not self.enabled or self.model is None:
            # Return base results if reranking disabled
            return list(candidates)[: self.top_k]

        candidates = list(candidates)[: self.top_k]

        # prepare pairs for cross-encoder: (query, doc_text)
        pairs = [(query, d.page_content) for d in candidates]

        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_docs = [doc for doc, _ in scored[: self.final_k]]
        return top_docs
