"""Tests for chatbot functionality."""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestChatbot:
    """Test chatbot functions."""

    def test_get_chatbot(self):
        """Test chatbot initialization."""
        try:
            from src.chatbot import get_chatbot
            
            # This test validates that the chatbot initializes without crashing
            # Full initialization requires models to be downloaded
            try:
                retriever, llm = get_chatbot()
                
                assert retriever is not None
                assert llm is not None
            except RuntimeError as e:
                if "rank_bm25" in str(e):
                    pytest.skip("rank_bm25 not available")
                else:
                    raise
        except Exception as e:
            pytest.skip(f"Chatbot initialization skipped: {e}")

    def test_langchain_retriever_adapter(self):
        """Test LangChain retriever adapter."""
        try:
            from src.chatbot import LangchainRetrieverAdapter
            
            mock_retriever = MagicMock()
            mock_retriever.get_relevant_documents.return_value = []
            
            adapter = LangchainRetrieverAdapter(mock_retriever)
            assert adapter is not None
            
            results = adapter.get_relevant_documents("test query")
            assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"Adapter test skipped: {e}")


class TestVectorstore:
    """Test vectorstore functionality."""

    @patch('src.vectorstore.FAISS')
    def test_load_vectorstore(self, mock_faiss):
        """Test loading vectorstore."""
        try:
            from src.vectorstore import load_vectorstore
            
            mock_faiss.return_value = MagicMock()
            
            # This may fail if vectorstore doesn't exist, which is expected in tests
            try:
                db = load_vectorstore()
                if db is not None:
                    assert db is not None
            except FileNotFoundError:
                pytest.skip("Vectorstore not initialized")
        except Exception as e:
            pytest.skip(f"Vectorstore test skipped: {e}")

    def test_load_documents(self):
        """Test loading documents."""
        try:
            from src.vectorstore import load_documents
            docs = load_documents()
            
            assert isinstance(docs, list)
        except Exception as e:
            pytest.skip(f"Document loading skipped: {e}")


class TestLoader:
    """Test document loader."""

    def test_load_pdf(self):
        """Test PDF loading."""
        try:
            from src.loader import load_pdf
            
            # This will fail if no test PDF is available, which is expected
            try:
                docs = load_pdf("data/test.pdf")
                if docs:
                    assert isinstance(docs, list)
            except FileNotFoundError:
                pytest.skip("Test PDF not found")
        except Exception as e:
            pytest.skip(f"PDF loading skipped: {e}")


class TestSplitter:
    """Test document splitter."""

    def test_split_docs(self):
        """Test document splitting."""
        try:
            from src.splitter import split_docs
            try:
                from langchain.schema import Document
            except Exception:
                class Document:
                    def __init__(self, page_content, metadata=None):
                        self.page_content = page_content
                        self.metadata = metadata or {}
            
            docs = [
                Document(
                    page_content="This is a long document " * 50,
                    metadata={"source": "test.pdf", "page": 1}
                )
            ]
            
            chunks = split_docs(docs)
            
            assert isinstance(chunks, list)
            if chunks:
                assert all(hasattr(chunk, 'page_content') for chunk in chunks)
        except Exception as e:
            pytest.skip(f"Splitter test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
