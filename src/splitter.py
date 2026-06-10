from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)
