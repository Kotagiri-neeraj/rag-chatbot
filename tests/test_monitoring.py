"""Tests for monitoring and logging."""

import pytest
import json
import tempfile
from pathlib import Path
from src.monitoring import (
    PerformanceMonitor,
    QueryLogger,
    MetricsCollector,
    timed,
    log_exception
)


class TestPerformanceMonitor:
    """Test performance monitoring."""

    def test_record_query(self):
        """Test query recording."""
        monitor = PerformanceMonitor()
        monitor.record_query("test query", duration=0.5, status="success")
        
        assert len(monitor.metrics["queries"]) == 1
        assert monitor.metrics["total_requests"] == 1
        query_metric = monitor.metrics["queries"][0]
        assert query_metric["query"] == "test query"
        assert query_metric["duration_ms"] == 500

    def test_record_retrieval(self):
        """Test retrieval recording."""
        monitor = PerformanceMonitor()
        monitor.record_retrieval(
            query="test",
            num_docs=5,
            duration=0.1,
            avg_relevance_score=0.85
        )
        
        assert len(monitor.metrics["retrievals"]) == 1
        ret_metric = monitor.metrics["retrievals"][0]
        assert ret_metric["num_docs"] == 5
        assert ret_metric["avg_relevance_score"] == 0.85

    def test_record_generation(self):
        """Test generation recording."""
        monitor = PerformanceMonitor()
        monitor.record_generation(answer_length=150, duration=0.4)
        
        assert len(monitor.metrics["generations"]) == 1
        gen_metric = monitor.metrics["generations"][0]
        assert gen_metric["answer_length"] == 150

    def test_get_stats(self):
        """Test statistics computation."""
        monitor = PerformanceMonitor()
        monitor.record_query("q1", duration=0.1)
        monitor.record_query("q2", duration=0.2)
        
        stats = monitor.get_stats()
        
        assert stats["total_requests"] == 2
        assert stats["query_stats"]["mean_ms"] == 150
        assert stats["query_stats"]["min_ms"] == 100
        assert stats["query_stats"]["max_ms"] == 200

    def test_save_metrics(self):
        """Test metrics saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor()
            monitor.record_query("test", duration=0.1)
            
            filepath = Path(tmpdir) / "metrics.json"
            monitor.save_metrics(str(filepath))
            
            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
                assert len(data["queries"]) == 1


class TestQueryLogger:
    """Test query logging."""

    def test_log_query(self):
        """Test query logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "queries.jsonl"
            logger = QueryLogger(str(log_file))
            
            logger.log_query(
                query="test query",
                retrieved_docs=["doc1", "doc2"],
                generated_answer="answer",
                duration=0.5
            )
            
            assert log_file.exists()
            with open(log_file) as f:
                line = f.readline()
                entry = json.loads(line)
                assert entry["query"] == "test query"
                assert entry["num_retrieved_docs"] == 2

    def test_log_query_with_metadata(self):
        """Test query logging with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "queries.jsonl"
            logger = QueryLogger(str(log_file))
            
            logger.log_query(
                query="test",
                retrieved_docs=["doc1"],
                generated_answer="answer",
                duration=0.5,
                metadata={"model": "flan-t5", "version": "1.0"}
            )
            
            with open(log_file) as f:
                entry = json.loads(f.readline())
                assert entry["metadata"]["model"] == "flan-t5"


class TestMetricsCollector:
    """Test metrics collection."""

    def test_log_rag_cycle(self):
        """Test RAG cycle logging."""
        collector = MetricsCollector()
        
        collector.log_rag_cycle(
            query="test query",
            retrieved_docs=["doc1", "doc2"],
            generated_answer="test answer",
            total_duration=0.6,
            retrieval_duration=0.2,
            generation_duration=0.4
        )
        
        assert collector.performance_monitor.metrics["total_requests"] == 1
        assert len(collector.performance_monitor.metrics["retrievals"]) == 1
        assert len(collector.performance_monitor.metrics["generations"]) == 1

    def test_get_report(self):
        """Test report generation."""
        collector = MetricsCollector()
        collector.log_rag_cycle(
            query="test",
            retrieved_docs=["doc1"],
            generated_answer="answer",
            total_duration=0.5,
            retrieval_duration=0.2,
            generation_duration=0.3
        )
        
        report = collector.get_report()
        
        assert "generated_at" in report
        assert "performance_stats" in report
        assert "log_files" in report


class TestDecorators:
    """Test decorator functions."""

    def test_timed_decorator(self):
        """Test timed decorator."""
        @timed
        def slow_function():
            import time
            time.sleep(0.01)
            return "result"
        
        result = slow_function()
        assert result == "result"

    def test_timed_decorator_exception(self):
        """Test timed decorator with exception."""
        @timed
        def failing_function():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            failing_function()

    def test_log_exception_decorator(self):
        """Test log exception decorator."""
        @log_exception
        def failing_function():
            raise RuntimeError("test error")
        
        with pytest.raises(RuntimeError):
            failing_function()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
