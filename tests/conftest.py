"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile

try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "AI_Guide.pdf", "page": 1}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "AI_Guide.pdf", "page": 2}
        ),
        Document(
            page_content="Natural language processing helps computers understand text.",
            metadata={"source": "NLP_Guide.pdf", "page": 1}
        ),
        Document(
            page_content="Transformers are powerful models for sequence tasks.",
            metadata={"source": "Transformers_Guide.pdf", "page": 5}
        ),
    ]


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_embeddings_model():
    """Create a mock embeddings model."""
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3] * 100  # 300-dim vector
    mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3] * 100] * 10
    return mock_model


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = {
        "output_text": "This is a sample response from the LLM."
    }
    return mock_llm


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = [
        Document(page_content="Retrieved document 1"),
        Document(page_content="Retrieved document 2"),
    ]
    return mock_retriever


@pytest.fixture
def evaluation_test_data():
    """Sample data for evaluation tests."""
    return {
        "query": "What is artificial intelligence?",
        "retrieved_docs": [
            "AI is the simulation of human intelligence by machines.",
            "Machine learning is a subset of AI."
        ],
        "generated_answer": "Artificial intelligence (AI) is the field of computer science focused on creating intelligent machines that can simulate human intelligence.",
        "ground_truth_docs": [
            "AI is the simulation of human intelligence by machines."
        ],
        "ground_truth_answer": "AI is artificial intelligence, the field of creating intelligent machines."
    }


@pytest.fixture
def monitoring_metrics_collector():
    """Create a metrics collector for testing."""
    from src.monitoring import MetricsCollector
    return MetricsCollector()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    try:
        import asyncio
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    yield loop
    loop.close()


# Markers for test categorization
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring model downloads"
    )
