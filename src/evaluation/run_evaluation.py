"""
Evaluation runner for AskAWS.

Pipeline:
  1. Load golden_set.yaml (questions + ground truth)
  2. Run each question through the multi-agent graph
  3. Collect: question, retrieved contexts, generated answer, ground truth
  4. Save raw results to data/evaluation_runs/<timestamp>/raw_results.json
     for downstream RAGAS scoring.

This step does NOT score — it just collects. Scoring is in score_with_ragas.py.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.graph import ask

PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLDEN_SET = PROJECT_ROOT / "src" / "evaluation" / "golden_set.yaml"
EVAL_ROOT = PROJECT_ROOT / "data" / "evaluation_runs"


def load_golden_set():
    """Flatten the YAML structure into a list of question dicts."""
    with open(GOLDEN_SET, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = []
    for service, questions in data.items():
        for q in questions:
            items.append({
                "service": service,
                "question": q["question"],
                "ground_truth": q["ground_truth"].strip(),
            })
    return items


def run_one(item: dict) -> dict:
    """Run one question through the graph and capture the data RAGAS needs."""
    state = ask(item["question"])

    # RAGAS expects: question, answer, contexts (list[str]), ground_truth
    chunks = state.get("retrieved_chunks", [])
    contexts = [c.page_content for c in chunks]

    return {
        "service": item["service"],
        "question": item["question"],
        "answer": state.get("final_answer", ""),
        "contexts": contexts,
        "ground_truth": item["ground_truth"],
        "route_decision": state.get("route_decision", ""),
        "critic_verdict": state.get("critic_verdict", ""),
        "critic_feedback": state.get("critic_feedback", ""),
        "n_chunks_retrieved": len(chunks),
        "best_distance": float(min(state.get("retrieved_distances", [1.0]))) if state.get("retrieved_distances") else None,
    }


def main():
    print("=" * 70)
    print("AskAWS Evaluation — Phase 1: Collecting Answers")
    print("=" * 70)

    items = load_golden_set()
    print(f"Loaded {len(items)} questions from golden set")

    # Output folder for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EVAL_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Output: {out_dir}\n")

    results = []
    start = time.time()
    for i, item in enumerate(items, 1):
        print(f"[{i:2d}/{len(items)}] [{item['service']:10s}] {item['question'][:60]}...")
        try:
            result = run_one(item)
            results.append(result)
            print(f"            -> route={result['route_decision']:8s} "
                  f"critic={result['critic_verdict']:5s} "
                  f"chunks={result['n_chunks_retrieved']}")
        except Exception as e:
            print(f"            -> FAILED: {e}")
            results.append({**item, "answer": "", "contexts": [], "error": str(e)})
        time.sleep(0.3)  # gentle pacing

    elapsed = time.time() - start
    out_file = out_dir / "raw_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "n_questions": len(items),
            "elapsed_seconds": elapsed,
            "results": results,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Raw results saved: {out_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
