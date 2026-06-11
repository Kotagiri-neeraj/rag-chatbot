"""Tests for evaluation metrics."""

import pytest
from src.evaluation import (
    RetrieverEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    RetrievalMetrics,
    GenerationMetrics
)


class TestRetrieverEvaluator:
    """Test retriever evaluation metrics."""

    def test_compute_ndcg(self):
        """Test NDCG computation."""
        relevance_scores = [1.0, 1.0, 0.0, 0.0, 0.0]
        ndcg = RetrieverEvaluator.compute_ndcg(relevance_scores, k=5)
        assert 0 <= ndcg <= 1.0
        assert ndcg > 0  # Should have some score for relevant items

    def test_compute_ndcg_empty(self):
        """Test NDCG with empty scores."""
        assert RetrieverEvaluator.compute_ndcg([], k=5) == 0.0

    def test_compute_mrr(self):
        """Test MRR computation."""
        relevance_scores = [0.0, 1.0, 1.0]
        mrr = RetrieverEvaluator.compute_mrr(relevance_scores)
        assert mrr == 0.5  # First relevant at position 1 (0-indexed)

    def test_compute_mrr_first_position(self):
        """Test MRR with first position relevant."""
        relevance_scores = [1.0, 0.0]
        mrr = RetrieverEvaluator.compute_mrr(relevance_scores)
        assert mrr == 1.0

    def test_compute_mrr_no_relevant(self):
        """Test MRR with no relevant items."""
        relevance_scores = [0.0, 0.0, 0.0]
        mrr = RetrieverEvaluator.compute_mrr(relevance_scores)
        assert mrr == 0.0

    def test_compute_precision_at_k(self):
        """Test Precision@k computation."""
        relevance_scores = [1.0, 1.0, 0.0, 0.0]
        precision = RetrieverEvaluator.compute_precision_at_k(relevance_scores, k=5)
        assert precision == 0.4  # 2 relevant out of 5

    def test_compute_precision_at_k_perfect(self):
        """Test Precision@k with perfect retrieval."""
        relevance_scores = [1.0, 1.0, 1.0]
        precision = RetrieverEvaluator.compute_precision_at_k(relevance_scores, k=3)
        assert precision == 1.0

    def test_compute_map(self):
        """Test MAP computation."""
        relevance_scores = [1.0, 0.0, 1.0]
        map_score = RetrieverEvaluator.compute_map(relevance_scores)
        assert 0 <= map_score <= 1.0

    def test_evaluate(self):
        """Test full retriever evaluation."""
        retrieved = ["doc1", "doc2", "doc3"]
        ground_truth = ["doc1", "doc2"]
        
        metrics = RetrieverEvaluator.evaluate(retrieved, ground_truth)
        
        assert isinstance(metrics, RetrievalMetrics)
        assert 0 <= metrics.ndcg_5 <= 1.0
        assert 0 <= metrics.mrr <= 1.0
        assert 0 <= metrics.precision_5 <= 1.0


class TestGenerationEvaluator:
    """Test generation evaluation metrics."""

    def test_compute_length_ratio(self):
        """Test length ratio computation."""
        generated = "This is a test"
        reference = "This is a test"
        ratio = GenerationEvaluator.compute_length_ratio(generated, reference)
        assert ratio == 1.0

    def test_compute_bleu_score(self):
        """Test BLEU score computation."""
        generated = "the cat is on the mat"
        reference = "the cat is on the mat"
        bleu = GenerationEvaluator.compute_bleu_score(generated, reference)
        assert bleu == 1.0  # Perfect match

    def test_compute_bleu_partial(self):
        """Test BLEU score with partial match."""
        generated = "the cat"
        reference = "the cat is on the mat"
        bleu = GenerationEvaluator.compute_bleu_score(generated, reference)
        assert 0 < bleu <= 1.0

    def test_compute_rouge_1(self):
        """Test ROUGE-1 computation."""
        generated = "the cat is on the mat"
        reference = "the cat is on the mat"
        rouge = GenerationEvaluator.compute_rouge_1(generated, reference)
        assert rouge == 1.0

    def test_compute_rouge_2(self):
        """Test ROUGE-2 computation."""
        generated = "the cat is on the mat"
        reference = "the cat is on the mat"
        rouge = GenerationEvaluator.compute_rouge_2(generated, reference)
        assert rouge == 1.0

    def test_compute_semantic_similarity(self):
        """Test semantic similarity computation."""
        generated = "What is the capital of France?"
        reference = "What is the capital of France?"
        sim = GenerationEvaluator.compute_semantic_similarity(generated, reference)
        assert 0 <= sim <= 1.0001  # Allow small floating-point precision error
        assert sim > 0.9  # Should be very similar

    def test_evaluate(self):
        """Test full generation evaluation."""
        generated = "The capital of France is Paris."
        reference = "Paris is the capital of France."
        
        metrics = GenerationEvaluator.evaluate(generated, reference)
        
        assert isinstance(metrics, GenerationMetrics)
        assert 0 <= metrics.semantic_similarity <= 1.0
        assert 0 <= metrics.bleu_score <= 1.0


class TestRAGEvaluator:
    """Test complete RAG evaluation."""

    def test_evaluate_rag_pipeline(self):
        """Test RAG pipeline evaluation."""
        evaluator = RAGEvaluator()
        
        result = evaluator.evaluate_rag_pipeline(
            query="What is AI?",
            retrieved_docs=["AI is artificial intelligence.", "AI learns from data."],
            generated_answer="AI is artificial intelligence that learns from data.",
            ground_truth_docs=["AI is artificial intelligence."],
            ground_truth_answer="AI is artificial intelligence."
        )
        
        assert "retrieval" in result
        assert "generation" in result
        assert result["query"] == "What is AI?"

    def test_aggregate_metrics(self):
        """Test metric aggregation."""
        eval_results = [
            {
                "retrieval": {"ndcg_5": 0.8, "mrr": 0.9},
                "generation": {"bleu": 0.7}
            },
            {
                "retrieval": {"ndcg_5": 0.9, "mrr": 0.8},
                "generation": {"bleu": 0.8}
            }
        ]
        
        aggregated = RAGEvaluator.aggregate_metrics(eval_results)
        
        assert aggregated["num_samples"] == 2
        assert "retrieval" in aggregated
        assert "generation" in aggregated
        assert abs(aggregated["retrieval"]["ndcg_5"]["mean"] - 0.85) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
