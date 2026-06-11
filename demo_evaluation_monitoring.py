#!/usr/bin/env python3
"""
Comprehensive example demonstrating evaluation, monitoring, and testing.
Run this script to see all features in action.
"""

import sys
import json
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import RAGEvaluator, RetrieverEvaluator, GenerationEvaluator
from src.monitoring import MetricsCollector, PerformanceMonitor


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_retrieval_evaluation():
    """Demonstrate retrieval evaluation."""
    print_section("1. Retrieval Quality Evaluation")
    
    evaluator = RetrieverEvaluator()
    
    # Simulate retrieval results
    retrieved_docs = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Python is a programming language",
        "Transformers are powerful models",
        "Web development uses JavaScript"
    ]
    
    ground_truth = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks"
    ]
    
    metrics = evaluator.evaluate(retrieved_docs, ground_truth, k=5)
    
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. {doc}")
    
    print("\nGround Truth:")
    for i, doc in enumerate(ground_truth, 1):
        print(f"  {i}. {doc}")
    
    print("\nRetrievalMetrics:")
    print(f"  NDCG@5:      {metrics.ndcg_5:.4f}")
    print(f"  NDCG@10:     {metrics.ndcg_10:.4f}")
    print(f"  MRR:         {metrics.mrr:.4f}")
    print(f"  Precision@5: {metrics.precision_5:.4f}")
    print(f"  Precision@10:{metrics.precision_10:.4f}")
    print(f"  Recall:      {metrics.recall:.4f}")
    print(f"  MAP:         {metrics.map_score:.4f}")


def demo_generation_evaluation():
    """Demonstrate generation evaluation."""
    print_section("2. Generation Quality Evaluation")
    
    evaluator = GenerationEvaluator()
    
    generated = "Machine learning is a subset of artificial intelligence that learns from data."
    reference = "Machine learning is a branch of AI that can learn from examples."
    
    print(f"Generated: {generated}")
    print(f"Reference: {reference}")
    
    metrics = evaluator.evaluate(generated, reference)
    
    print("\nGeneration Metrics:")
    print(f"  Semantic Similarity: {metrics.semantic_similarity:.4f}")
    print(f"  Length Ratio:        {metrics.length_ratio:.4f}")
    print(f"  BLEU Score:          {metrics.bleu_score:.4f}")
    print(f"  ROUGE-1:             {metrics.rouge_1:.4f}")
    print(f"  ROUGE-2:             {metrics.rouge_2:.4f}")


def demo_rag_evaluation():
    """Demonstrate complete RAG evaluation."""
    print_section("3. Complete RAG Pipeline Evaluation")
    
    rag_evaluator = RAGEvaluator()
    
    # Example RAG results
    results = []
    
    eval_examples = [
        {
            "query": "What is machine learning?",
            "retrieved": ["ML is a subset of AI", "ML learns from data"],
            "generated": "Machine learning is a type of AI that learns patterns from data.",
            "ground_truth_docs": ["ML is a subset of AI"],
            "ground_truth_answer": "Machine learning is artificial intelligence that learns from data."
        },
        {
            "query": "Explain neural networks",
            "retrieved": ["Neural networks use layers", "They process information"],
            "generated": "Neural networks are computing systems inspired by the brain.",
            "ground_truth_docs": ["Neural networks use layers"],
            "ground_truth_answer": "Neural networks are interconnected layers of nodes."
        },
        {
            "query": "What is deep learning?",
            "retrieved": ["Deep learning uses many layers", "It's part of ML"],
            "generated": "Deep learning is a subset of machine learning using neural networks.",
            "ground_truth_docs": ["Deep learning uses many layers"],
            "ground_truth_answer": "Deep learning uses deep neural networks."
        }
    ]
    
    for example in eval_examples:
        result = rag_evaluator.evaluate_rag_pipeline(
            query=example["query"],
            retrieved_docs=example["retrieved"],
            generated_answer=example["generated"],
            ground_truth_docs=example["ground_truth_docs"],
            ground_truth_answer=example["ground_truth_answer"]
        )
        results.append(result)
        
        print(f"\nQuery: {example['query']}")
        print(f"Retrieval NDCG@5: {result['retrieval']['ndcg_5']:.4f}")
        print(f"Generation Similarity: {result['generation']['semantic_similarity']:.4f}")
    
    # Aggregate results
    print("\n" + "-" * 70)
    print("Aggregated Metrics Across All Queries:")
    aggregated = RAGEvaluator.aggregate_metrics(results)
    
    print(f"\nRetrieval Metrics:")
    for metric, stats in aggregated["retrieval"].items():
        print(f"  {metric}:")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    
    print(f"\nGeneration Metrics:")
    for metric, stats in aggregated["generation"].items():
        print(f"  {metric}:")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")


def demo_monitoring():
    """Demonstrate monitoring and logging."""
    print_section("4. Performance Monitoring")
    
    collector = MetricsCollector()
    
    # Simulate multiple RAG cycles
    print("Simulating RAG cycles...")
    
    for i in range(3):
        query = f"Test query {i+1}"
        
        # Simulate retrieval
        retrieval_start = time.time()
        time.sleep(0.05)  # Simulate retrieval time
        retrieval_duration = time.time() - retrieval_start
        
        # Simulate generation
        gen_start = time.time()
        time.sleep(0.1)  # Simulate generation time
        gen_duration = time.time() - gen_start
        
        total_duration = retrieval_duration + gen_duration
        
        collector.log_rag_cycle(
            query=query,
            retrieved_docs=[f"Doc {j}" for j in range(5)],
            generated_answer=f"Answer to {query}",
            total_duration=total_duration,
            retrieval_duration=retrieval_duration,
            generation_duration=gen_duration,
            metadata={"iteration": i+1}
        )
    
    # Get statistics
    print("\nPerformance Statistics:")
    stats = collector.performance_monitor.get_stats()
    
    print(f"Total Requests: {stats['total_requests']}")
    
    if stats['query_stats']:
        print("\nQuery Processing:")
        print(f"  Mean: {stats['query_stats']['mean_ms']:.2f}ms")
        print(f"  Min: {stats['query_stats']['min_ms']:.2f}ms")
        print(f"  Max: {stats['query_stats']['max_ms']:.2f}ms")
    
    if stats['retrieval_stats']:
        print("\nRetrieval:")
        print(f"  Mean: {stats['retrieval_stats']['mean_ms']:.2f}ms")
        print(f"  Min: {stats['retrieval_stats']['min_ms']:.2f}ms")
        print(f"  Max: {stats['retrieval_stats']['max_ms']:.2f}ms")
    
    if stats['generation_stats']:
        print("\nGeneration:")
        print(f"  Mean: {stats['generation_stats']['mean_ms']:.2f}ms")
        print(f"  Min: {stats['generation_stats']['min_ms']:.2f}ms")
        print(f"  Max: {stats['generation_stats']['max_ms']:.2f}ms")
    
    # Generate report
    print("\n" + "-" * 70)
    print("Performance Report:")
    report = collector.get_report()
    print(json.dumps(report, indent=2))


def print_testing_guide():
    """Print a guide for running tests."""
    print_section("5. Testing Guide")
    
    guide = """
Available Test Commands:

1. Run all tests:
   pytest tests/ -v

2. Run specific test file:
   pytest tests/test_evaluation.py -v

3. Run with coverage:
   pytest tests/ --cov=src --cov-report=html

4. Run specific test:
   pytest tests/test_evaluation.py::TestRetrieverEvaluator::test_compute_ndcg -v

5. Run tests matching pattern:
   pytest tests/ -k "test_retrieval" -v

6. Run tests with markers:
   pytest tests/ -m "unit" -v
   pytest tests/ -m "integration" -v

7. Run tests with detailed output:
   pytest tests/ -vv --tb=long

8. Generate test report:
   pytest tests/ --html=report.html

CI/CD Pipeline:
   The project includes GitHub Actions CI pipeline that runs:
   - Unit tests with coverage
   - Linting (flake8, black, isort)
   - Security checks (bandit, safety)
   - Build verification

Test Files:
   - tests/test_evaluation.py:    Evaluation metrics
   - tests/test_monitoring.py:    Monitoring and logging
   - tests/test_retriever.py:     Retrieval components
   - tests/test_chatbot.py:       Chatbot functionality

Coverage Target: 70%+
"""
    print(guide)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  RAG CHATBOT - EVALUATION, MONITORING & TESTING DEMO")
    print("=" * 70)
    
    try:
        demo_retrieval_evaluation()
        demo_generation_evaluation()
        demo_rag_evaluation()
        demo_monitoring()
        print_testing_guide()
        
        print("\n" + "=" * 70)
        print("  DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        print("\nNext Steps:")
        print("1. Run tests: pytest tests/ -v")
        print("2. Check logs: logs/chatbot.log and logs/queries.jsonl")
        print("3. Review metrics: logs/metrics_*.json")
        print("4. Run CI locally: pytest && flake8 && bandit")
        
    except Exception as e:
        print(f"\n Error during demo: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
