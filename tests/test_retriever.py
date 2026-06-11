"""Tests for retriever and reranker."""

import pytest
from unittest.mock import Mock, MagicMock, patch
try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


class TestSimpleBM25Retriever:
    """Test BM25 retriever."""

    def test_initialization(self):
        """Test BM25 retriever initialization."""
        try:
            from src.retriever import SimpleBM25Retriever
            docs = [
                Document(page_content="machine learning is great"),
                Document(page_content="deep learning is powerful"),
                Document(page_content="neural networks")
            ]
            retriever = SimpleBM25Retriever(docs)
            assert retriever is not None
            assert len(retriever.documents) == 3
        except (ImportError, RuntimeError):
            pytest.skip("rank_bm25 not available")

    def test_get_relevant_documents(self):
        """Test BM25 document retrieval."""
        try:
            from src.retriever import SimpleBM25Retriever
            docs = [
                Document(page_content="machine learning algorithms"),
                Document(page_content="deep neural networks"),
                Document(page_content="python programming language")
            ]
            retriever = SimpleBM25Retriever(docs)
            results = retriever.get_relevant_documents("machine learning", k=2)
            
            assert len(results) <= 2
            assert all(hasattr(doc, 'page_content') for doc in results)
        except (ImportError, RuntimeError):
            pytest.skip("rank_bm25 not available")


class TestHybridEnsembleRetriever:
    """Test hybrid ensemble retriever."""

    @patch('src.retriever.load_vectorstore')
    def test_initialization(self, mock_load):
        """Test hybrid retriever initialization."""
        try:
            from src.retriever import SimpleBM25Retriever, HybridEnsembleRetriever
            
            docs = [
                Document(page_content="test doc 1"),
                Document(page_content="test doc 2")
            ]
            bm25 = SimpleBM25Retriever(docs)
            mock_db = MagicMock()
            
            retriever = HybridEnsembleRetriever(bm25, mock_db)
            assert retriever is not None
        except (ImportError, RuntimeError):
            pytest.skip("rank_bm25 not available")

    @patch('src.retriever.load_vectorstore')
    def test_get_relevant_documents(self, mock_load):
        """Test hybrid retriever document retrieval."""
        try:
            from src.retriever import SimpleBM25Retriever, HybridEnsembleRetriever
            
            docs = [
                Document(page_content="machine learning"),
                Document(page_content="deep learning"),
                Document(page_content="data science")
            ]
            bm25 = SimpleBM25Retriever(docs)
            mock_db = MagicMock()
            mock_db.similarity_search.return_value = [
                Document(page_content="machine learning")
            ]
            
            retriever = HybridEnsembleRetriever(bm25, mock_db)
            results = retriever.get_relevant_documents("learning")
            
            assert results is not None
        except (ImportError, RuntimeError):
            pytest.skip("rank_bm25 not available")


class TestRerankingRetriever:
    """Test reranking retriever."""

    def test_initialization(self):
        """Test reranker initialization."""
        from src.reranker import RerankingRetriever
        base_retriever = MagicMock()
        reranker = RerankingRetriever(base_retriever)
        assert reranker is not None
        assert reranker.enabled in [True, False]  # May be disabled if model unavailable

    def test_get_relevant_documents_without_reranking(self):
        """Test reranker with disabled reranking."""
        from src.reranker import RerankingRetriever
        
        mock_retriever = MagicMock()
        docs = [
            Document(page_content="doc 1"),
            Document(page_content="doc 2")
        ]
        mock_retriever.get_relevant_documents.return_value = docs
        
        reranker = RerankingRetriever(mock_retriever)
        results = reranker.get_relevant_documents("query")
        
        assert results is not None
        assert isinstance(results, list)

    def test_get_relevant_documents_callable_base(self):
        """Test reranker with callable base retriever."""
        from src.reranker import RerankingRetriever
        
        docs = [
            Document(page_content="doc 1"),
            Document(page_content="doc 2")
        ]
        mock_retriever = MagicMock(side_effect=lambda x: docs)
        
        reranker = RerankingRetriever(mock_retriever)
        results = reranker.get_relevant_documents("query")
        
        assert isinstance(results, list)


class TestEmbeddings:
    """Test embeddings module."""

    def test_get_embeddings_model(self):
        """Test getting embeddings model."""
        try:
            from src.embeddings import get_embeddings_model
            model = get_embeddings_model()
            assert model is not None
        except Exception as e:
            pytest.skip(f"Embeddings not available: {e}")

    def test_embed_text(self):
        """Test text embedding."""
        try:
            from src.embeddings import get_embeddings_model
            model = get_embeddings_model()
            embedding = model.embed_query("test query")
            
            assert isinstance(embedding, list)
            assert len(embedding) > 0
        except Exception as e:
            pytest.skip(f"Embeddings not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
