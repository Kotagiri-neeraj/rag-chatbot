from langchain.embeddings.base import Embeddings
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


def get_embeddings():
    return TfidfEmbeddings()
