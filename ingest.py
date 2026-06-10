from pathlib import Path
try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

from src.loader import load_pdf
from src.splitter import split_docs
from src.vectorstore import create_vectorstore

data_dir = Path("data")
pdf_files = sorted(data_dir.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found in data/. Add at least one PDF such as company_policy.pdf."
    )

docs = []
for pdf_file in pdf_files:
    print(f"Loading {pdf_file}")
    loaded = load_pdf(str(pdf_file))
    for idx, doc in enumerate(loaded, start=1):
        # try to extract a page number from the loader metadata when available,
        # otherwise fall back to the enumeration index
        page_num = None
        try:
            meta = getattr(doc, "metadata", {}) or {}
            page_num = meta.get("page") or meta.get("page_number") or meta.get("pagenumber")
        except Exception:
            page_num = None

        if page_num is None:
            page_num = idx

        # store only the filename for display (e.g. employee_handbook.pdf)
        docs.append(Document(page_content=doc.page_content, metadata={
            "source": pdf_file.name,
            "page": page_num
        }))

chunks = split_docs(docs)
create_vectorstore(chunks)
print("Done")
