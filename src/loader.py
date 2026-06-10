try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.schema import Document as _LangchainDocument

    def load_pdf(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

except Exception:
    # Fallback simple PDF loader using pypdf when langchain_community isn't available
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader
        except Exception:
            raise RuntimeError("Install 'pypdf' or 'PyPDF2' to enable PDF loading.")

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    def load_pdf(file_path):
        reader = PdfReader(file_path)
        docs = []
        # PyPDF2 / pypdf have slightly different page access, but .pages works
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                # older PyPDF2 may use .extractText()
                try:
                    text = page.extractText() or ""
                except Exception:
                    text = ""
            docs.append(Document(page_content=text, metadata={"page": i}))
        return docs
