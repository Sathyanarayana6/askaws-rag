"""
CLI entrypoint for the AskAWS multi-agent RAG system.

Usage:
    python scripts/ask_multiagent.py            # interactive REPL
    python scripts/ask_multiagent.py --debug    # show full agent trace
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import ask


def print_response(state: dict, debug: bool = False):
    """Pretty-print a final state from the graph."""
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(state.get("final_answer", "(no answer produced)"))

    # Router decision
    route = state.get("route_decision", "?")
    if route == "refuse":
        print("\n[Status] Refused at router")
    else:
        # Citations
        chunks = state.get("retrieved_chunks", [])
        distances = state.get("retrieved_distances", [])
        if chunks:
            print("\n" + "-" * 70)
            print("SOURCES")
            print("-" * 70)
            for i, (doc, dist) in enumerate(zip(chunks, distances), start=1):
                service = doc.metadata.get("service", "?").upper()
                title = doc.metadata.get("title", "?")
                source = doc.metadata.get("source_url", "?")
                print(f"  [{i}] [{service}] {title}")
                print(f"      {source}")
                print(f"      (distance: {dist:.3f})")

        # Critic verdict
        verdict = state.get("critic_verdict")
        if verdict:
            print(f"\n[Critic verdict]    {verdict}")
            print(f"[Critic feedback]   {state.get('critic_feedback', '')}")

    # Debug trace
    if debug:
        print("\n" + "-" * 70)
        print("AGENT TRACE")
        print("-" * 70)
        for line in state.get("trace", []):
            print(f"  {line}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AskAWS Multi-Agent RAG")
    parser.add_argument("--debug", action="store_true", help="Show agent trace")
    args = parser.parse_args()

    print("=" * 70)
    print("AskAWS Multi-Agent RAG")
    print("Agents: Router -> Retrieval -> Generator -> Critic")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 70)

    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        print("\n... routing ...")
        try:
            state = ask(question)
        except Exception as e:
            print(f"\nERROR: {e}\n")
            continue

        print_response(state, debug=args.debug)

    print("Goodbye.")


if __name__ == "__main__":
    main()