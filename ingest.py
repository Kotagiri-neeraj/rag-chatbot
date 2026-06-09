from pathlib import Path
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
    docs.extend(load_pdf(str(pdf_file)))

chunks = split_docs(docs)
create_vectorstore(chunks)
print("Done")
