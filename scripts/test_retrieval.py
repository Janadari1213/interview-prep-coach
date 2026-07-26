"""
Retrieval Evaluation Test Script for Interview Preparation Coach.

Executes 5 benchmark interview queries against ChromaDB retriever,
printing top-3 retrieved chunks and source metadata for manual relevance logging.
"""

import sys
import io
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

from rag.retriever import get_relevant_chunks


SAMPLE_QUERIES = [
    "What is the STAR method?",
    "Difference between abstract class and interface",
    "How to answer 'tell me about a time you failed'",
    "Common DevOps interview questions",
    "How to solve a two-sum coding problem"
]


def evaluate_retriever():
    """Run benchmark queries against ChromaDB vector store."""
    print("=" * 70, flush=True)
    print("🔍 RAG RETRIEVER EVALUATION BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    for idx, query in enumerate(SAMPLE_QUERIES, start=1):
        print(f"\nQuery {idx}: '{query}'", flush=True)
        print("-" * 50, flush=True)

        chunks = get_relevant_chunks(query=query, top_k=3)

        if not chunks:
            print("  [Warning] No matching chunks returned. Please run python rag/ingest.py first.", flush=True)
        else:
            for rank, c in enumerate(chunks, start=1):
                src = c.get("source", "unknown")
                text_snippet = c.get("text", "").replace("\n", " ")[:120]
                print(f"  Result #{rank} [{src}]: {text_snippet}...", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("✅ RETRIEVAL EVALUATION COMPLETED", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    evaluate_retriever()
