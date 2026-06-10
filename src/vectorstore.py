import pickle
from pathlib import Path

from src.embeddings import get_embeddings
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    try:
        from langchain.vectorstores import FAISS
    except Exception:
        # Provide a minimal FAISS-compatible wrapper if langchain FAISS classes are not available.
        try:
            import faiss
            import numpy as np
        except Exception:
            faiss = None

        class FAISS:
            def __init__(self, index, docs, embeddings):
                self.index = index
                self.docs = docs
                self.embeddings = embeddings

            @classmethod
            def from_documents(cls, documents, embeddings):
                # embed documents
                texts = [d.page_content for d in documents]
                vectors = embeddings.embed_documents(texts)
                npv = np.array(vectors).astype('float32')
                if faiss is not None:
                    dim = npv.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(npv)
                else:
                    index = (npv,)
                return cls(index=index, docs=documents, embeddings=embeddings)

            def save_local(self, folder_path: str):
                import os, pickle
                os.makedirs(folder_path, exist_ok=True)
                # save docs and index numpy arrays
                with open(os.path.join(folder_path, 'docs.pkl'), 'wb') as f:
                    pickle.dump(self.docs, f)
                if faiss is not None:
                    faiss.write_index(self.index, os.path.join(folder_path, 'index.faiss'))
                else:
                    # save numpy vectors
                    np.save(os.path.join(folder_path, 'vectors.npy'), np.array(self.embeddings.embed_documents([d.page_content for d in self.docs])))

            @classmethod
            def load_local(cls, folder_path: str, embeddings, allow_dangerous_deserialization=False):
                import os, pickle
                with open(os.path.join(folder_path, 'docs.pkl'), 'rb') as f:
                    docs = pickle.load(f)
                if faiss is not None and os.path.exists(os.path.join(folder_path, 'index.faiss')):
                    index = faiss.read_index(os.path.join(folder_path, 'index.faiss'))
                else:
                    vectors_path = os.path.join(folder_path, 'vectors.npy')
                    if os.path.exists(vectors_path):
                        npv = np.load(vectors_path)
                        index = (npv,)
                    else:
                        index = None
                return cls(index=index, docs=docs, embeddings=embeddings)

            def as_retriever(self):
                class Retriever:
                    def __init__(self, parent):
                        self.parent = parent

                    def get_relevant_documents(self, query):
                        # use embeddings to embed query and do similarity search
                        qv = np.array(self.parent.embeddings.embed_query(query)).astype('float32')
                        if faiss is not None and self.parent.index is not None:
                            D, I = self.parent.index.search(np.expand_dims(qv, axis=0), 5)
                            return [self.parent.docs[i] for i in I[0] if i < len(self.parent.docs)]
                        else:
                            # brute-force
                            npv = self.parent.index[0]
                            sims = ((npv - qv) ** 2).sum(axis=1)
                            idx = sims.argsort()[:5]
                            return [self.parent.docs[i] for i in idx]

                return Retriever(self)

            def similarity_search_with_score(self, query, k=5):
                qv = np.array(self.embeddings.embed_query(query)).astype('float32')
                if faiss is not None and self.index is not None:
                    D, I = self.index.search(np.expand_dims(qv, axis=0), k)
                    return [(self.docs[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.docs)]
                else:
                    npv = self.index[0]
                    sims = ((npv - qv) ** 2).sum(axis=1)
                    idx = sims.argsort()[:k]
                    return [(self.docs[i], float(sims[i])) for i in idx]

        # end FAISS wrapper

# simple Document fallback
try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

VECTORSTORE_DIR = Path("vectorstore")
EMBEDDINGS_FILE = VECTORSTORE_DIR / "embeddings.pkl"

def create_vectorstore(chunks):
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(VECTORSTORE_DIR))
    VECTORSTORE_DIR.mkdir(exist_ok=True, parents=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    return db


def load_vectorstore():
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore folder '{VECTORSTORE_DIR}' does not exist. Run ingest.py first."
        )

    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)
    else:
        index_file = VECTORSTORE_DIR / "index.pkl"
        if not index_file.exists():
            raise FileNotFoundError(
                f"Neither embeddings file '{EMBEDDINGS_FILE}' nor index metadata '{index_file}' were found. "
                "Recreate the vectorstore with ingest.py."
            )
        with open(index_file, "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        texts = []
        for doc_id in index_to_docstore_id.values():
            doc = docstore.search(doc_id)
            if doc is None:
                continue
            texts.append(doc.page_content)
        embeddings = get_embeddings()
        embeddings.embed_documents(texts)
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings, f)

    db = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db


def load_documents():
    """Return the original list of Documents saved in the FAISS docstore (if available).
    This reads the serialized index.pkl that `FAISS.save_local` creates.
    """
    index_file = VECTORSTORE_DIR / "index.pkl"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Index metadata '{index_file}' not found. Run ingest.py to (re)create the vectorstore."
        )

    with open(index_file, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    docs = []
    # index_to_docstore_id maps index positions to stored IDs
    for doc_id in index_to_docstore_id.values():
        try:
            stored = docstore.search(doc_id)
        except Exception:
            stored = None
        if stored is None:
            continue
        # stored is expected to be a Document-like object
        if isinstance(stored, Document):
            docs.append(stored)
        else:
            # try to build a Document
            text = getattr(stored, "page_content", None) or str(stored)
            meta = getattr(stored, "metadata", {}) or {}
            docs.append(Document(page_content=text, metadata=meta))

    return docs
