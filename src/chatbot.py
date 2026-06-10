# from langchain_community.llms import HuggingFacePipeline
# from src.vectorstore import load_vectorstore
# from transformers import pipeline

# def get_chatbot():
#     db = load_vectorstore()
#     retriever = db.as_retriever()

#     text_generation = pipeline(
#         "text-generation",
#         model="google/flan-t5-small",
#         do_sample=False,
#         max_new_tokens=256,
#         device=-1
#     )
#     llm = HuggingFacePipeline(pipeline=text_generation)
#     return retriever, llm


from src.retriever import build_hybrid_retriever
from src.reranker import RerankingRetriever
from transformers import pipeline

try:
    from langchain.schema import BaseRetriever
except Exception:
    BaseRetriever = object


class LangchainRetrieverAdapter(BaseRetriever):
    """Adapter to expose a wrapped retriever as a LangChain BaseRetriever-compatible object."""
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def get_relevant_documents(self, query: str):
        return self.wrapped.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str):
        if hasattr(self.wrapped, "aget_relevant_documents"):
            return await self.wrapped.aget_relevant_documents(query)
        # fallback to sync
        return self.wrapped.get_relevant_documents(query)


def get_chatbot():
    # build a hybrid BM25 + FAISS retriever with smaller candidate sizes for speed
    base = build_hybrid_retriever(k_bm25=5, k_faiss=5, weights=(0.5, 0.5))

    # wrap with a cross-encoder reranker: top 6 -> rerank -> top 2
    reranking = RerankingRetriever(base_retriever=base, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=6, final_k=2)

    # wrap reranker in an adapter that subclasses LangChain's BaseRetriever
    retriever_adapter = LangchainRetrieverAdapter(reranking)

    # detect GPU availability and set device
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = "cpu"

    # Use tokenizer + seq2seq model directly to avoid unsupported pipeline task errors
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Move model to device and put in eval mode
    try:
        model.to(device)
    except Exception:
        pass
    model.eval()

    def generate(text: str, max_new_tokens: int = 512) -> str:
        """Generate response from flan-t5-small model."""
        try:
            import torch as _torch
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            # Move inputs to model device
            try:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            except Exception:
                pass
            
            # Generate with parameters optimized for longer, detailed answers
            with _torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_length=50,  # Force at least 50 tokens
                    length_penalty=2.0,  # Encourage longer sequences
                    num_beams=1,  # Use greedy for consistency
                    temperature=0.8,
                    early_stopping=False
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            result = decoded[0].strip() if decoded else ""
            return result if result else "(Unable to generate response)"
        except Exception as e:
            import traceback
            error_msg = str(e)
            return f"(Error: {error_msg[:100]})"

    return retriever_adapter, generate