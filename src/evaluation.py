"""
Evaluation metrics for RAG chatbot retrieval and generation quality.
Includes:
  - Retrieval metrics (NDCG, MRR, Precision@K)
  - Generation metrics (semantic similarity, length metrics)
  - Full evaluation pipeline
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    ndcg_5: float
    ndcg_10: float
    mrr: float
    precision_5: float
    precision_10: float
    recall: float
    map_score: float


@dataclass
class GenerationMetrics:
    """Container for generation evaluation metrics."""
    semantic_similarity: float
    length_ratio: float
    bleu_score: float
    rouge_1: float
    rouge_2: float


class RetrieverEvaluator:
    """Evaluate retrieval quality against ground truth."""

    @staticmethod
    def compute_ndcg(relevance_scores: List[float], k: int = 5) -> float:
        """Compute Normalized Discounted Cumulative Gain at k."""
        if not relevance_scores:
            return 0.0
        
        # DCG@k
        dcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(relevance_scores[:k])
        )
        
        # IDCG@k (ideal ranking)
        ideal_relevance = sorted(relevance_scores, reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(ideal_relevance)
        )
        
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_mrr(relevance_scores: List[float]) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def compute_precision_at_k(relevance_scores: List[float], k: int = 5) -> float:
        """Compute Precision@k (fraction of top-k that are relevant)."""
        if not relevance_scores or k == 0:
            return 0.0
        top_k = relevance_scores[:k]
        relevant = sum(1 for rel in top_k if rel > 0)
        return relevant / k

    @staticmethod
    def compute_recall(relevance_scores: List[float], threshold: float = 0.5) -> float:
        """Compute Recall (fraction of all relevant items retrieved)."""
        total_relevant = sum(1 for rel in relevance_scores if rel >= threshold)
        retrieved_relevant = sum(1 for rel in relevance_scores if rel >= threshold)
        return retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

    @staticmethod
    def compute_map(relevance_scores: List[float], k: int = 10) -> float:
        """Compute Mean Average Precision."""
        if not relevance_scores:
            return 0.0
        
        score = 0.0
        num_relevant = 0
        
        for i, rel in enumerate(relevance_scores[:k]):
            if rel > 0:
                num_relevant += 1
                precision = num_relevant / (i + 1)
                score += precision
        
        return score / min(len([r for r in relevance_scores if r > 0]), k) if num_relevant > 0 else 0.0

    @classmethod
    def evaluate(
        cls, 
        retrieved_docs: List[str],
        ground_truth: List[str],
        k: int = 5
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_docs: List of retrieved document strings
            ground_truth: List of relevant ground truth documents
            k: Cutoff for metrics
            
        Returns:
            RetrievalMetrics object with computed scores
        """
        # Simple relevance scoring: 1 if doc is in ground truth, 0 otherwise
        relevance_scores = [
            1.0 if doc in ground_truth else 0.0
            for doc in retrieved_docs
        ]
        
        return RetrievalMetrics(
            ndcg_5=cls.compute_ndcg(relevance_scores, k=5),
            ndcg_10=cls.compute_ndcg(relevance_scores, k=10),
            mrr=cls.compute_mrr(relevance_scores),
            precision_5=cls.compute_precision_at_k(relevance_scores, k=5),
            precision_10=cls.compute_precision_at_k(relevance_scores, k=10),
            recall=cls.compute_recall(relevance_scores),
            map_score=cls.compute_map(relevance_scores)
        )


class GenerationEvaluator:
    """Evaluate generation quality."""

    @staticmethod
    def compute_semantic_similarity(generated: str, reference: str) -> float:
        """Compute semantic similarity using simple token overlap."""
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb1 = model.encode(generated, convert_to_tensor=True)
            emb2 = model.encode(reference, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(emb1, emb2).item()
            return float(sim)
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}. Falling back to token overlap.")
            # Fallback to token overlap
            gen_tokens = set(generated.lower().split())
            ref_tokens = set(reference.lower().split())
            if not gen_tokens or not ref_tokens:
                return 0.0
            return len(gen_tokens & ref_tokens) / len(gen_tokens | ref_tokens)

    @staticmethod
    def compute_length_ratio(generated: str, reference: str) -> float:
        """Compute ratio of generated to reference length."""
        gen_len = len(generated.split())
        ref_len = len(reference.split())
        if ref_len == 0:
            return 0.0
        return min(gen_len, ref_len) / max(gen_len, ref_len)

    @staticmethod
    def compute_bleu_score(generated: str, reference: str) -> float:
        """Compute simplified BLEU-1 score (unigram precision)."""
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        if not gen_tokens:
            return 0.0
        
        matches = sum(1 for token in gen_tokens if token in ref_tokens)
        return matches / len(gen_tokens)

    @staticmethod
    def compute_rouge_1(generated: str, reference: str) -> float:
        """Compute ROUGE-1 (unigram recall)."""
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return 0.0
        
        matches = len(gen_tokens & ref_tokens)
        return matches / len(ref_tokens)

    @staticmethod
    def compute_rouge_2(generated: str, reference: str) -> float:
        """Compute ROUGE-2 (bigram recall)."""
        def get_bigrams(text: str) -> set:
            tokens = text.lower().split()
            return set(zip(tokens[:-1], tokens[1:]))
        
        gen_bigrams = get_bigrams(generated)
        ref_bigrams = get_bigrams(reference)
        
        if not ref_bigrams:
            return 0.0
        
        matches = len(gen_bigrams & ref_bigrams)
        return matches / len(ref_bigrams)

    @classmethod
    def evaluate(
        cls,
        generated: str,
        reference: str
    ) -> GenerationMetrics:
        """
        Evaluate generation quality.
        
        Args:
            generated: Generated text
            reference: Reference/ground truth text
            
        Returns:
            GenerationMetrics object with computed scores
        """
        return GenerationMetrics(
            semantic_similarity=cls.compute_semantic_similarity(generated, reference),
            length_ratio=cls.compute_length_ratio(generated, reference),
            bleu_score=cls.compute_bleu_score(generated, reference),
            rouge_1=cls.compute_rouge_1(generated, reference),
            rouge_2=cls.compute_rouge_2(generated, reference)
        )


class RAGEvaluator:
    """Complete RAG pipeline evaluator."""

    def __init__(self):
        self.retriever_evaluator = RetrieverEvaluator()
        self.generation_evaluator = GenerationEvaluator()

    def evaluate_rag_pipeline(
        self,
        query: str,
        retrieved_docs: List[str],
        generated_answer: str,
        ground_truth_docs: List[str],
        ground_truth_answer: str
    ) -> Dict:
        """
        Evaluate complete RAG pipeline.
        
        Args:
            query: User query
            retrieved_docs: Documents retrieved by retriever
            generated_answer: Answer generated by LLM
            ground_truth_docs: Ground truth relevant documents
            ground_truth_answer: Ground truth answer
            
        Returns:
            Dict with all metrics
        """
        retrieval_metrics = self.retriever_evaluator.evaluate(
            retrieved_docs, ground_truth_docs
        )
        
        generation_metrics = self.generation_evaluator.evaluate(
            generated_answer, ground_truth_answer
        )
        
        return {
            "query": query,
            "retrieval": {
                "ndcg_5": retrieval_metrics.ndcg_5,
                "ndcg_10": retrieval_metrics.ndcg_10,
                "mrr": retrieval_metrics.mrr,
                "precision_5": retrieval_metrics.precision_5,
                "precision_10": retrieval_metrics.precision_10,
                "recall": retrieval_metrics.recall,
                "map": retrieval_metrics.map_score
            },
            "generation": {
                "semantic_similarity": generation_metrics.semantic_similarity,
                "length_ratio": generation_metrics.length_ratio,
                "bleu": generation_metrics.bleu_score,
                "rouge_1": generation_metrics.rouge_1,
                "rouge_2": generation_metrics.rouge_2
            }
        }

    @staticmethod
    def aggregate_metrics(eval_results: List[Dict]) -> Dict:
        """Aggregate evaluation results across multiple samples."""
        if not eval_results:
            return {}
        
        metrics = {
            "retrieval": {},
            "generation": {},
            "num_samples": len(eval_results)
        }
        
        # Aggregate retrieval metrics
        retrieval_keys = eval_results[0]["retrieval"].keys()
        for key in retrieval_keys:
            values = [r["retrieval"][key] for r in eval_results]
            metrics["retrieval"][key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        # Aggregate generation metrics
        generation_keys = eval_results[0]["generation"].keys()
        for key in generation_keys:
            values = [r["generation"][key] for r in eval_results]
            metrics["generation"][key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return metrics


if __name__ == "__main__":
    # Example usage
    evaluator = RAGEvaluator()
    
    # Example evaluation
    result = evaluator.evaluate_rag_pipeline(
        query="What is the capital of France?",
        retrieved_docs=["Paris is the capital of France.", "France is in Europe."],
        generated_answer="Paris is the capital of France. It is located in central France.",
        ground_truth_docs=["Paris is the capital of France."],
        ground_truth_answer="Paris is the capital of France."
    )
    
    print("Evaluation Results:")
    import json
    print(json.dumps(result, indent=2))
