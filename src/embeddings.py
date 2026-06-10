try:
    from langchain.embeddings.base import Embeddings
except Exception:
    class Embeddings:
        pass


# Prefer HuggingFace/HF-transformers embeddings (BGE). Fall back to TF-IDF if unavailable.
def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5"):
    try:
        # langchain_huggingface provides a compatible Embeddings wrapper
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception:
        # lightweight TF-IDF fallback for environments without HF models
        from sklearn.feature_extraction.text import TfidfVectorizer

        class TfidfEmbeddings(Embeddings):
            def __init__(self):
                self.vectorizer = TfidfVectorizer()
                self.fitted = False

            def embed_documents(self, texts):
                if not self.fitted:
                    self.vectors = self.vectorizer.fit_transform(texts)
                    self.fitted = True
                return self.vectors.toarray().tolist()

            def embed_query(self, text):
                if not self.fitted:
                    raise ValueError("Embeddings model must be fit on documents before embedding queries.")
                return self.vectorizer.transform([text]).toarray()[0].tolist()

        return TfidfEmbeddings()
