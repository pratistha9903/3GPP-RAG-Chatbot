#!/usr/bin/env python3
"""Interactive CLI for the 3GPP KG-RAG chatbot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.rag_pipeline import KGRAGPipeline


def main():
    print("=" * 60)
    print("3GPP KG-RAG Chatbot (CLI)")
    print("Near-zero hallucination RAG for telecom standards")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop.\n")

    pipeline = KGRAGPipeline()
    print("Initializing pipeline...")
    pipeline.initialize()
    print("Ready!\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = pipeline.query(query)

        print(f"\n{'='*40}")
        if response.route:
            print(f"[Route: {response.route.intent.value} | Index: {response.route.index_hint}]")
        if response.rejected:
            print(f"[Rejected: {response.rejection_reason}]")
        print(f"\nAssistant: {response.answer}")

        if response.retrieved_chunks:
            print(f"\n--- Retrieved {len(response.retrieved_chunks)} chunks ---")
            for c in response.retrieved_chunks[:3]:
                print(f"  [{c.rank}] {c.citation()} (score: {c.score:.3f})")

        if response.retrieved_triples:
            print(f"\n--- Retrieved {len(response.retrieved_triples)} KG triples ---")
            for t in response.retrieved_triples[:3]:
                print(f"  {t.to_text()[:120]}")

        if response.verification_result:
            v = response.verification_result
            print(f"\n--- Verification: {v.verdict} (confidence: {v.confidence:.2f}) ---")

        print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
