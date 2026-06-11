"""
RAGAS (RAG Assessment) metrics implementation for RAG chatbot evaluation.
Provides production-grade metrics: Faithfulness, Context Precision, Answer Relevancy.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RAGASMetrics:
    """Container for RAGAS evaluation metrics."""
    faithfulness: float
    context_precision: float
    answer_relevancy: float
    ragas_score: float  # Weighted average of all metrics


class RAGASEvaluator:
    """
    RAGAS (RAG Assessment) evaluator for production-grade metrics.
    
    Implements:
    - Faithfulness: How much of the answer is grounded in the retrieved context
    - Context Precision: How many of the retrieved documents are relevant to the query
    - Answer Relevancy: How directly does the answer address the question
    """

    def __init__(self):
        self.embeddings_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Embeddings model not available: {e}. Using fallback methods.")

    def compute_faithfulness(
        self,
        answer: str,
        context: str,
        threshold: float = 0.5
    ) -> float:
        """
        Compute faithfulness score (0-1).
        
        Measures how much of the answer is grounded in the provided context.
        High faithfulness means the answer is supported by the context.
        
        Args:
            answer: Generated answer text
            context: Retrieved context text
            threshold: Similarity threshold for considering a sentence faithful
            
        Returns:
            Faithfulness score (0-1)
        """
        if not answer or not context:
            return 0.0
        
        # Split into sentences (simple approach)
        answer_sents = [s.strip() for s in answer.split('.') if s.strip()]
        context_sents = [s.strip() for s in context.split('.') if s.strip()]
        
        if not answer_sents:
            return 1.0  # Empty answer is perfectly faithful
        
        # Count sentences from answer that appear in context
        faithful_sents = 0
        for ans_sent in answer_sents:
            # Check if sentence or significant portion appears in context
            if any(ans_sent.lower() in ctx_sent.lower() or 
                   self._semantic_similarity(ans_sent, ctx_sent) > threshold
                   for ctx_sent in context_sents):
                faithful_sents += 1
        
        return faithful_sents / len(answer_sents)

    def compute_context_precision(
        self,
        query: str,
        retrieved_docs: List[str],
        ground_truth_docs: List[str]
    ) -> float:
        """
        Compute context precision score (0-1).
        
        Measures how many of the retrieved documents are relevant to the query.
        High precision means most retrieved docs are actually relevant.
        
        Args:
            query: User query
            retrieved_docs: Documents retrieved by the system
            ground_truth_docs: Ground truth relevant documents
            
        Returns:
            Context precision score (0-1)
        """
        if not retrieved_docs:
            return 0.0
        
        if not ground_truth_docs:
            return 0.0
        
        # Count how many retrieved docs are in ground truth
        relevant_count = 0
        for retrieved in retrieved_docs:
            if any(self._content_overlap(retrieved, gt) > 0.5 
                   for gt in ground_truth_docs):
                relevant_count += 1
        
        return relevant_count / len(retrieved_docs)

    def compute_answer_relevancy(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Compute answer relevancy score (0-1).
        
        Measures how well the answer addresses the query.
        High relevancy means the answer is directly related to the question.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            Answer relevancy score (0-1)
        """
        if not query or not answer:
            return 0.0
        
        # Compute similarity between query and answer
        similarity = self._semantic_similarity(query, answer)
        
        # Check keyword overlap
        query_keywords = set(query.lower().split())
        answer_words = set(answer.lower().split())
        keyword_overlap = len(query_keywords & answer_words) / len(query_keywords) if query_keywords else 0
        
        # Weighted combination
        return 0.6 * similarity + 0.4 * keyword_overlap

    def compute_ragas_score(
        self,
        faithfulness: float,
        context_precision: float,
        answer_relevancy: float,
        weights: Tuple[float, float, float] = (0.3, 0.3, 0.4)
    ) -> float:
        """
        Compute overall RAGAS score.
        
        Weighted average of faithfulness, context precision, and answer relevancy.
        
        Args:
            faithfulness: Faithfulness score (0-1)
            context_precision: Context precision score (0-1)
            answer_relevancy: Answer relevancy score (0-1)
            weights: Weights for each metric (must sum to 1.0)
            
        Returns:
            RAGAS score (0-1)
        """
        return (
            weights[0] * faithfulness +
            weights[1] * context_precision +
            weights[2] * answer_relevancy
        )

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        retrieved_docs: List[str],
        ground_truth_docs: Optional[List[str]] = None
    ) -> RAGASMetrics:
        """
        Complete RAGAS evaluation.
        
        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context (concatenated)
            retrieved_docs: List of retrieved documents
            ground_truth_docs: Optional ground truth documents for context precision
            
        Returns:
            RAGASMetrics object with all scores
        """
        faithfulness = self.compute_faithfulness(answer, context)
        
        context_precision = 1.0
        if ground_truth_docs:
            context_precision = self.compute_context_precision(
                query, retrieved_docs, ground_truth_docs
            )
        
        answer_relevancy = self.compute_answer_relevancy(query, answer)
        
        ragas_score = self.compute_ragas_score(
            faithfulness, context_precision, answer_relevancy
        )
        
        return RAGASMetrics(
            faithfulness=faithfulness,
            context_precision=context_precision,
            answer_relevancy=answer_relevancy,
            ragas_score=ragas_score
        )

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if self.embeddings_model is not None:
            try:
                from sentence_transformers import util
                emb1 = self.embeddings_model.encode(text1, convert_to_tensor=True)
                emb2 = self.embeddings_model.encode(text2, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(emb1, emb2).item()
                return float(sim)
            except Exception:
                pass
        
        # Fallback: token overlap
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _content_overlap(doc1: str, doc2: str) -> float:
        """Compute content overlap between two documents."""
        tokens1 = set(doc1.lower().split())
        tokens2 = set(doc2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        return intersection / min(len(tokens1), len(tokens2))

    @staticmethod
    def aggregate_ragas_scores(
        eval_results: List[RAGASMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate RAGAS scores across multiple evaluations.
        
        Args:
            eval_results: List of RAGASMetrics objects
            
        Returns:
            Dictionary with aggregated statistics
        """
        if not eval_results:
            return {}
        
        faithfulness_scores = [r.faithfulness for r in eval_results]
        precision_scores = [r.context_precision for r in eval_results]
        relevancy_scores = [r.answer_relevancy for r in eval_results]
        ragas_scores = [r.ragas_score for r in eval_results]
        
        return {
            "faithfulness": {
                "mean": float(np.mean(faithfulness_scores)),
                "std": float(np.std(faithfulness_scores)),
                "min": float(np.min(faithfulness_scores)),
                "max": float(np.max(faithfulness_scores))
            },
            "context_precision": {
                "mean": float(np.mean(precision_scores)),
                "std": float(np.std(precision_scores)),
                "min": float(np.min(precision_scores)),
                "max": float(np.max(precision_scores))
            },
            "answer_relevancy": {
                "mean": float(np.mean(relevancy_scores)),
                "std": float(np.std(relevancy_scores)),
                "min": float(np.min(relevancy_scores)),
                "max": float(np.max(relevancy_scores))
            },
            "ragas_score": {
                "mean": float(np.mean(ragas_scores)),
                "std": float(np.std(ragas_scores)),
                "min": float(np.min(ragas_scores)),
                "max": float(np.max(ragas_scores))
            },
            "num_samples": len(eval_results)
        }


if __name__ == "__main__":
    evaluator = RAGASEvaluator()
    
    result = evaluator.evaluate(
        query="What is machine learning?",
        answer="Machine learning is a subset of AI that learns from data.",
        context="Machine learning is a subset of artificial intelligence. ML uses algorithms to learn from data.",
        retrieved_docs=["ML is a subset of AI", "ML learns from data"],
        ground_truth_docs=["ML is a subset of AI"]
    )
    
    print("RAGAS Evaluation Results:")
    print(f"  Faithfulness: {result.faithfulness:.4f}")
    print(f"  Context Precision: {result.context_precision:.4f}")
    print(f"  Answer Relevancy: {result.answer_relevancy:.4f}")
    print(f"  RAGAS Score: {result.ragas_score:.4f}")
