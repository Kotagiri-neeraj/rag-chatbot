#!/usr/bin/env python3
"""
View and analyze logs and metrics from the RAG chatbot.
Usage:
    python view_metrics.py --logs           # Show recent logs
    python view_metrics.py --metrics        # Show metrics
    python view_metrics.py --queries        # Show recent queries
    python view_metrics.py --all           # Show all
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse


def format_iso_time(iso_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_str


def view_logs(log_file: str = "logs/chatbot.log", lines: int = 20):
    """View recent log entries."""
    log_path = Path(log_file)
    
    print(f"\n{'=' * 70}")
    print(f"  Recent Logs (Last {lines} entries from {log_file})")
    print(f"{'=' * 70}\n")
    
    if not log_path.exists():
        print("❌ No logs found. Run the chatbot first to generate logs.")
        return
    
    try:
        with open(log_path) as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                try:
                    entry = json.loads(line)
                    timestamp = entry.get("timestamp", "")
                    level = entry.get("level", "INFO")
                    event = entry.get("event", "")
                    
                    print(f"[{format_iso_time(timestamp)}] {level}: {event}")
                    
                    # Print other fields
                    for key, value in entry.items():
                        if key not in ["timestamp", "level", "event"]:
                            print(f"  {key}: {value}")
                except:
                    print(line.strip())
    except Exception as e:
        print(f"❌ Error reading logs: {e}")


def view_queries(query_log: str = "logs/queries.jsonl", count: int = 10):
    """View recent queries."""
    query_path = Path(query_log)
    
    print(f"\n{'=' * 70}")
    print(f"  Recent Queries (Last {count} from {query_log})")
    print(f"{'=' * 70}\n")
    
    if not query_path.exists():
        print("❌ No query logs found. Run the chatbot first to generate queries.")
        return
    
    try:
        with open(query_path) as f:
            entries = [json.loads(line) for line in f]
        
        # Show last 'count' entries
        for entry in entries[-count:]:
            print(f"Query: {entry.get('query', '')[:60]}...")
            print(f"  Time: {format_iso_time(entry.get('timestamp', ''))}")
            print(f"  Duration: {entry.get('duration_seconds', 0):.2f}s")
            print(f"  Retrieved Docs: {entry.get('num_retrieved_docs', 0)}")
            print(f"  Answer Length: {entry.get('answer_length', 0)} chars")
            
            if entry.get('metadata'):
                print(f"  Metadata: {entry['metadata']}")
            print()
    except Exception as e:
        print(f"❌ Error reading queries: {e}")


def view_metrics(metrics_dir: str = "logs"):
    """View latest metrics."""
    metrics_path = Path(metrics_dir)
    
    print(f"\n{'=' * 70}")
    print(f"  Performance Metrics from {metrics_dir}")
    print(f"{'=' * 70}\n")
    
    if not metrics_path.exists():
        print("❌ No metrics directory found.")
        return
    
    # Find latest metrics file
    metric_files = list(metrics_path.glob("metrics_*.json"))
    
    if not metric_files:
        print("❌ No metrics files found.")
        return
    
    latest = sorted(metric_files)[-1]
    
    print(f"Latest Metrics File: {latest.name}\n")
    
    try:
        with open(latest) as f:
            data = json.load(f)
        
        # Print stats
        print(f"Total Requests: {len(data.get('queries', []))}")
        print(f"Total Retrievals: {len(data.get('retrievals', []))}")
        print(f"Total Generations: {len(data.get('generations', []))}")
        
        # Retrieve timing stats
        if data.get('queries'):
            durations = [q['duration_ms'] for q in data['queries']]
            print(f"\nQuery Timing:")
            print(f"  Mean: {sum(durations)/len(durations):.2f}ms")
            print(f"  Min: {min(durations):.2f}ms")
            print(f"  Max: {max(durations):.2f}ms")
        
        # Retrieval stats
        if data.get('retrievals'):
            ret_times = [r['duration_ms'] for r in data['retrievals']]
            print(f"\nRetrieval Timing:")
            print(f"  Mean: {sum(ret_times)/len(ret_times):.2f}ms")
            print(f"  Min: {min(ret_times):.2f}ms")
            print(f"  Max: {max(ret_times):.2f}ms")
        
        # Generation stats
        if data.get('generations'):
            gen_times = [g['duration_ms'] for g in data['generations']]
            gen_lengths = [g['answer_length'] for g in data['generations']]
            print(f"\nGeneration Timing:")
            print(f"  Mean: {sum(gen_times)/len(gen_times):.2f}ms")
            print(f"  Min: {min(gen_times):.2f}ms")
            print(f"  Max: {max(gen_times):.2f}ms")
            print(f"\nGenerated Answer Lengths:")
            print(f"  Mean: {sum(gen_lengths)/len(gen_lengths):.0f} chars")
            print(f"  Min: {min(gen_lengths)}")
            print(f"  Max: {max(gen_lengths)}")
            
    except Exception as e:
        print(f"❌ Error reading metrics: {e}")


def generate_report():
    """Generate comprehensive report."""
    print(f"\n{'=' * 70}")
    print("  RAG CHATBOT MONITORING REPORT")
    print(f"{'=' * 70}\n")
    
    view_queries(count=5)
    view_metrics()
    view_logs(lines=10)
    
    print(f"\n{'=' * 70}")
    print("  END OF REPORT")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="View RAG chatbot metrics and logs"
    )
    parser.add_argument(
        '--logs', action='store_true',
        help='Show recent logs'
    )
    parser.add_argument(
        '--metrics', action='store_true',
        help='Show metrics'
    )
    parser.add_argument(
        '--queries', action='store_true',
        help='Show recent queries'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Show all (default if no option specified)'
    )
    parser.add_argument(
        '--count', type=int, default=10,
        help='Number of items to show'
    )
    
    args = parser.parse_args()
    
    # If no options specified, show report
    if not (args.logs or args.metrics or args.queries):
        args.all = True
    
    try:
        if args.logs or args.all:
            view_logs(lines=args.count)
        
        if args.metrics or args.all:
            view_metrics()
        
        if args.queries or args.all:
            view_queries(count=args.count)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
