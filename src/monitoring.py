"""
Monitoring and logging infrastructure for RAG chatbot.
Provides structured logging, metrics collection, and performance tracking.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
from pathlib import Path
import structlog

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Setup standard logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "chatbot.log"),
        logging.StreamHandler()
    ]
)

logger = structlog.get_logger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        self.metrics = {
            "queries": [],
            "retrievals": [],
            "generations": [],
            "total_requests": 0
        }

    def record_query(self, query: str, duration: float, status: str = "success"):
        """Record query processing metrics."""
        self.metrics["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "duration_ms": duration * 1000,
            "status": status
        })
        self.metrics["total_requests"] += 1
        
        logger.info(
            "query_processed",
            query_length=len(query.split()),
            duration_ms=duration * 1000,
            status=status
        )

    def record_retrieval(
        self,
        query: str,
        num_docs: int,
        duration: float,
        avg_relevance_score: float = 0.0
    ):
        """Record retrieval metrics."""
        self.metrics["retrievals"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_docs": num_docs,
            "duration_ms": duration * 1000,
            "avg_relevance_score": avg_relevance_score
        })
        
        logger.info(
            "retrieval_completed",
            num_documents=num_docs,
            duration_ms=duration * 1000,
            avg_score=avg_relevance_score
        )

    def record_generation(
        self,
        answer_length: int,
        duration: float,
        model: str = "flan-t5-small"
    ):
        """Record generation metrics."""
        self.metrics["generations"].append({
            "timestamp": datetime.now().isoformat(),
            "answer_length": answer_length,
            "duration_ms": duration * 1000,
            "model": model
        })
        
        logger.info(
            "generation_completed",
            answer_length=answer_length,
            duration_ms=duration * 1000,
            model=model
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        import numpy as np
        
        stats = {
            "total_requests": self.metrics["total_requests"],
            "query_stats": self._compute_stats(
                [q["duration_ms"] for q in self.metrics["queries"]]
            ) if self.metrics["queries"] else {},
            "retrieval_stats": self._compute_stats(
                [r["duration_ms"] for r in self.metrics["retrievals"]]
            ) if self.metrics["retrievals"] else {},
            "generation_stats": self._compute_stats(
                [g["duration_ms"] for g in self.metrics["generations"]]
            ) if self.metrics["generations"] else {}
        }
        return stats

    @staticmethod
    def _compute_stats(values: list) -> Dict[str, float]:
        """Compute statistics for a list of values."""
        if not values:
            return {}
        import numpy as np
        return {
            "mean_ms": float(np.mean(values)),
            "std_ms": float(np.std(values)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "median_ms": float(np.median(values))
        }

    def save_metrics(self, filepath: Optional[str] = None):
        """Save metrics to JSON file."""
        if filepath is None:
            filepath = LOG_DIR / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info("metrics_saved", filepath=str(filepath))

    def log_stats(self):
        """Log current statistics."""
        stats = self.get_stats()
        logger.info("performance_stats", stats=stats)
        return stats


class QueryLogger:
    """Log detailed query information."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or LOG_DIR / "queries.jsonl"

    def log_query(
        self,
        query: str,
        retrieved_docs: list,
        generated_answer: str,
        duration: float,
        metadata: Optional[Dict] = None
    ):
        """Log complete query information."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_retrieved_docs": len(retrieved_docs),
            "retrieved_docs_preview": [d[:100] for d in retrieved_docs[:3]],
            "answer_length": len(generated_answer),
            "duration_seconds": duration,
            "metadata": metadata or {}
        }
        
        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info("query_logged", query_id=hash(query), duration=duration)


def timed(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(
                f"{func.__name__}_completed",
                duration_ms=duration * 1000,
                status="success"
            )
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(
                f"{func.__name__}_failed",
                duration_ms=duration * 1000,
                error=str(e),
                status="error"
            )
            raise
    return wrapper


def log_exception(func: Callable) -> Callable:
    """Decorator to log exceptions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(
                f"{func.__name__}_exception",
                exc_info=True,
                error_type=type(e).__name__
            )
            raise
    return wrapper


class MetricsCollector:
    """Collect and export metrics."""

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.query_logger = QueryLogger()

    def log_rag_cycle(
        self,
        query: str,
        retrieved_docs: list,
        generated_answer: str,
        total_duration: float,
        retrieval_duration: float,
        generation_duration: float,
        metadata: Optional[Dict] = None
    ):
        """Log a complete RAG cycle."""
        self.query_logger.log_query(
            query=query,
            retrieved_docs=retrieved_docs,
            generated_answer=generated_answer,
            duration=total_duration,
            metadata={
                **(metadata or {}),
                "retrieval_duration": retrieval_duration,
                "generation_duration": generation_duration
            }
        )
        
        self.performance_monitor.record_query(
            query=query,
            duration=total_duration,
            status="success"
        )
        
        self.performance_monitor.record_retrieval(
            query=query,
            num_docs=len(retrieved_docs),
            duration=retrieval_duration
        )
        
        self.performance_monitor.record_generation(
            answer_length=len(generated_answer),
            duration=generation_duration
        )

    def get_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        stats = self.performance_monitor.get_stats()
        return {
            "generated_at": datetime.now().isoformat(),
            "performance_stats": stats,
            "log_files": {
                "queries": str(self.query_logger.log_file),
                "chatbot": str(LOG_DIR / "chatbot.log")
            }
        }

    def export_metrics(self, filepath: Optional[str] = None):
        """Export all metrics."""
        report = self.get_report()
        self.performance_monitor.save_metrics()
        logger.info("metrics_exported", filepath=filepath or str(LOG_DIR))
        return report


# Global instance
metrics_collector = MetricsCollector()


if __name__ == "__main__":
    # Example usage
    collector = MetricsCollector()
    
    # Log a sample RAG cycle
    collector.log_rag_cycle(
        query="What is machine learning?",
        retrieved_docs=[
            "Machine learning is a subset of AI.",
            "ML uses algorithms to learn from data."
        ],
        generated_answer="Machine learning is a subset of artificial intelligence that uses algorithms to learn patterns from data.",
        total_duration=0.523,
        retrieval_duration=0.123,
        generation_duration=0.400,
        metadata={"model": "flan-t5-small", "retrieval_type": "hybrid"}
    )
    
    # Get and log stats
    stats = collector.performance_monitor.log_stats()
    
    # Generate and print report
    report = collector.get_report()
    print(json.dumps(report, indent=2))
