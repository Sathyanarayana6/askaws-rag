"""Re-aggregate ragas_summary.json from existing scored_results.json (no re-scoring)."""

import json
import math
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_ROOT = PROJECT_ROOT / "data" / "evaluation_runs"


def fmt(v):
    if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and math.isnan(v)):
        return f"{v:.3f}"
    return "  n/a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run ID; defaults to latest")
    args = parser.parse_args()

    runs = sorted(EVAL_ROOT.glob("*"))
    run_dir = EVAL_ROOT / args.run if args.run else runs[-1]

    raw = json.loads((run_dir / "raw_results.json").read_text(encoding="utf-8"))
    scored = json.loads((run_dir / "scored_results.json").read_text(encoding="utf-8"))

    # Map question text -> service from raw_results
    q_to_service = {r["question"]: r["service"] for r in raw["results"]}

    METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Overall
    overall = {}
    for m in METRICS:
        vals = [
            row[m] for row in scored
            if m in row and isinstance(row[m], (int, float)) and not (isinstance(row[m], float) and math.isnan(row[m]))
        ]
        overall[m] = round(sum(vals) / len(vals), 4) if vals else None

    # Per service
    per_service = {}
    for row in scored:
        svc = q_to_service.get(row.get("user_input") or row.get("question"))
        if not svc:
            continue
        per_service.setdefault(svc, []).append(row)

    service_summary = {}
    for svc, rows in per_service.items():
        svc_metrics = {"n": len(rows)}
        for m in METRICS:
            vals = [
                r[m] for r in rows
                if m in r and isinstance(r[m], (int, float)) and not (isinstance(r[m], float) and math.isnan(r[m]))
            ]
            svc_metrics[m] = round(sum(vals) / len(vals), 4) if vals else None
        service_summary[svc] = svc_metrics

    summary = {
        "run_id": run_dir.name,
        "n_evaluated": len(scored),
        "metrics": overall,
        "per_service": service_summary,
        "rescored_at": datetime.now().isoformat(),
    }

    (run_dir / "ragas_summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 70)
    print(f"Run: {run_dir.name}")
    print("=" * 70)
    print(f"Overall ({summary['n_evaluated']} questions):")
    for m in METRICS:
        print(f"  {m:22s} {fmt(overall.get(m))}")
    print()
    print("Per service:")
    for svc, m in service_summary.items():
        print(f"  {svc:10s} (n={m['n']})  "
              f"faith={fmt(m.get('faithfulness'))}  "
              f"relev={fmt(m.get('answer_relevancy'))}  "
              f"prec ={fmt(m.get('context_precision'))}  "
              f"recall={fmt(m.get('context_recall'))}")
    print()
    print(f"Updated: {run_dir / 'ragas_summary.json'}")


if __name__ == "__main__":
    main()
