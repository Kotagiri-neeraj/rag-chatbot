import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from src.embeddings import get_embeddings

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
