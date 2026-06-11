"""
Langfuse integration for LLM observability and tracing.
Provides production-grade tracing for RAG pipelines.
"""

import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """
    Langfuse integration for RAG pipeline tracing.
    
    Langfuse is a production LLM observability platform that tracks:
    - Traces: Complete RAG pipeline executions
    - Spans: Individual component executions (retrieval, generation)
    - Events: Important system events
    - Metrics: Performance and quality metrics
    """

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize Langfuse tracer.
        
        Args:
            api_key: Langfuse API key (from env if not provided)
            endpoint: Langfuse endpoint (optional custom endpoint)
        """
        self.enabled = False
        self.client = None
        self.current_trace = None
        
        try:
            # Try to import and initialize Langfuse
            try:
                from langfuse import Langfuse
            except ImportError:
                logger.warning("Langfuse not installed. Tracing disabled. Install with: pip install langfuse")
                return
            
            # Initialize client
            if api_key:
                self.client = Langfuse(api_key=api_key, host=endpoint)
            else:
                # Will use LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY env vars
                self.client = Langfuse()
            
            self.enabled = True
            logger.info("Langfuse tracer initialized successfully")
        except Exception as e:
            logger.warning(f"Langfuse initialization failed: {e}. Tracing disabled.")

    def start_trace(
        self,
        name: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new trace for a RAG pipeline execution.
        
        Args:
            name: Trace name (e.g., "rag_query_pipeline")
            input_data: Input to the pipeline
            metadata: Optional metadata
            
        Returns:
            Trace ID
        """
        if not self.enabled or not self.client:
            return "disabled"
        
        try:
            trace_id = f"trace_{datetime.now().timestamp()}"
            self.current_trace = self.client.trace(
                name=name,
                input=input_data,
                metadata=metadata or {},
                id=trace_id
            )
            logger.debug(f"Trace started: {trace_id}")
            return trace_id
        except Exception as e:
            logger.warning(f"Failed to start trace: {e}")
            return "error"

    def add_retrieval_span(
        self,
        query: str,
        retrieved_docs: List[str],
        retrieval_time: float,
        num_candidates: int = 0
    ):
        """
        Log retrieval component span.
        
        Args:
            query: User query
            retrieved_docs: Documents retrieved
            retrieval_time: Time taken for retrieval
            num_candidates: Number of candidates before reranking
        """
        if not self.enabled or not self.client:
            return
        
        try:
            span = self.client.span(
                name="retrieval",
                input={"query": query, "num_candidates": num_candidates},
                output={
                    "num_retrieved": len(retrieved_docs),
                    "docs_preview": [doc[:100] for doc in retrieved_docs[:3]]
                },
                metadata={
                    "retrieval_time_ms": retrieval_time * 1000,
                    "document_count": len(retrieved_docs)
                }
            )
            logger.debug("Retrieval span logged")
        except Exception as e:
            logger.warning(f"Failed to log retrieval span: {e}")

    def add_generation_span(
        self,
        query: str,
        answer: str,
        generation_time: float,
        model_name: str = "flan-t5-small",
        tokens_generated: int = 0
    ):
        """
        Log generation component span.
        
        Args:
            query: User query
            answer: Generated answer
            generation_time: Time taken for generation
            model_name: Name of the model used
            tokens_generated: Number of tokens generated
        """
        if not self.enabled or not self.client:
            return
        
        try:
            span = self.client.span(
                name="generation",
                input={"query": query},
                output={"answer": answer[:200]},
                metadata={
                    "generation_time_ms": generation_time * 1000,
                    "model": model_name,
                    "tokens_generated": tokens_generated,
                    "answer_length": len(answer)
                }
            )
            logger.debug("Generation span logged")
        except Exception as e:
            logger.warning(f"Failed to log generation span: {e}")

    def add_reranking_span(
        self,
        input_count: int,
        output_count: int,
        reranking_time: float,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Log reranking component span.
        
        Args:
            input_count: Number of input documents
            output_count: Number of output documents
            reranking_time: Time taken for reranking
            model_name: Name of reranker model
        """
        if not self.enabled or not self.client:
            return
        
        try:
            span = self.client.span(
                name="reranking",
                input={"num_candidates": input_count},
                output={"num_reranked": output_count},
                metadata={
                    "reranking_time_ms": reranking_time * 1000,
                    "model": model_name,
                    "compression_ratio": output_count / input_count if input_count > 0 else 0
                }
            )
            logger.debug("Reranking span logged")
        except Exception as e:
            logger.warning(f"Failed to log reranking span: {e}")

    def end_trace(
        self,
        output: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        End the current trace.
        
        Args:
            output: Final output/answer
            metrics: Optional evaluation metrics
        """
        if not self.enabled or not self.client:
            return
        
        try:
            if self.current_trace:
                self.current_trace.end(
                    output=output,
                    metadata=metrics or {}
                )
                logger.debug("Trace ended")
        except Exception as e:
            logger.warning(f"Failed to end trace: {e}")

    def log_evaluation(
        self,
        trace_id: str,
        faithfulness: float,
        context_precision: float,
        answer_relevancy: float,
        ragas_score: float
    ):
        """
        Log evaluation metrics to a trace.
        
        Args:
            trace_id: ID of the trace to update
            faithfulness: Faithfulness score
            context_precision: Context precision score
            answer_relevancy: Answer relevancy score
            ragas_score: Overall RAGAS score
        """
        if not self.enabled or not self.client:
            return
        
        try:
            self.client.score(
                name="ragas_metrics",
                value=ragas_score,
                trace_id=trace_id,
                comment=json.dumps({
                    "faithfulness": faithfulness,
                    "context_precision": context_precision,
                    "answer_relevancy": answer_relevancy
                })
            )
            logger.debug(f"Evaluation metrics logged for trace {trace_id}")
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics: {e}")

    def flush(self):
        """Flush all pending traces to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
                logger.info("Langfuse traces flushed")
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse: {e}")


class LocalTracer:
    """
    Fallback local tracer when Langfuse is not available.
    Saves traces to JSON files for local analysis.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.traces = []

    def start_trace(
        self,
        name: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a local trace."""
        trace_id = f"trace_{datetime.now().timestamp()}"
        self.current_trace = {
            "id": trace_id,
            "name": name,
            "started_at": datetime.now().isoformat(),
            "input": input_data,
            "metadata": metadata or {},
            "spans": []
        }
        return trace_id

    def add_span(self, span_name: str, input_data: Dict, output_data: Dict, metadata: Dict):
        """Add a span to the current trace."""
        if hasattr(self, 'current_trace'):
            self.current_trace["spans"].append({
                "name": span_name,
                "input": input_data,
                "output": output_data,
                "metadata": metadata
            })

    def end_trace(self, output: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """End the current trace."""
        if hasattr(self, 'current_trace'):
            self.current_trace["ended_at"] = datetime.now().isoformat()
            self.current_trace["output"] = output
            self.current_trace["metrics"] = metrics or {}
            self.traces.append(self.current_trace)

    def save_traces(self, filepath: str = "logs/traces.jsonl"):
        """Save all traces to a JSONL file."""
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(exist_ok=True)
        
        with open(filepath, 'a') as f:
            for trace in self.traces:
                f.write(json.dumps(trace) + '\n')
        
        logger.info(f"Saved {len(self.traces)} traces to {filepath}")
        self.traces = []


# Global tracer instance
_tracer: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """Get or create global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer


if __name__ == "__main__":
    tracer = get_tracer()
    
    # Example usage
    trace_id = tracer.start_trace(
        name="rag_query",
        input_data={"query": "What is AI?"}
    )
    
    tracer.add_retrieval_span(
        query="What is AI?",
        retrieved_docs=["AI is...", "Machine learning is..."],
        retrieval_time=0.15
    )
    
    tracer.add_generation_span(
        query="What is AI?",
        answer="AI is artificial intelligence...",
        generation_time=0.35
    )
    
    tracer.end_trace(
        output={"answer": "AI is artificial intelligence..."},
        metrics={"latency_ms": 500}
    )
    
    tracer.flush()
