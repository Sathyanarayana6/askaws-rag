"""
Score RAGAS metrics on evaluation results.

Reads raw_results.json from an evaluation run, scores each question on:
  - faithfulness:       Is the answer supported by retrieved context?
  - answer_relevancy:   Does the answer address the question?
  - context_precision:  Were the right chunks retrieved?
  - context_recall:     Did retrieval get all the needed info?

RAGAS is configured to use Bedrock Claude + Titan embeddings — no OpenAI.

Usage:
    python src/evaluation/score_with_ragas.py
    python src/evaluation/score_with_ragas.py --run 20260518_072757
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import boto3
from dotenv import load_dotenv
from datasets import Dataset

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# LangChain wrappers (RAGAS expects LangChain-compatible LLM + embeddings)
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_ROOT = PROJECT_ROOT / "data" / "evaluation_runs"

GENERATION_MODEL = os.getenv("BEDROCK_GENERATION_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def latest_run_dir() -> Path:
    """Find the most recent evaluation run folder."""
    runs = sorted(EVAL_ROOT.glob("*"))
    if not runs:
        raise FileNotFoundError("No evaluation runs found. Run run_evaluation.py first.")
    return runs[-1]


def build_ragas_components():
    """Set up RAGAS to use Bedrock Claude + Titan instead of OpenAI default."""
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

    llm = ChatBedrock(
        client=bedrock,
        model_id=GENERATION_MODEL,
        model_kwargs={"temperature": 0.0, "max_tokens": 1024},
    )
    embeddings = BedrockEmbeddings(client=bedrock, model_id=EMBEDDING_MODEL)

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def filter_evaluable(results: list[dict]) -> list[dict]:
    """RAGAS needs contexts + answer. Skip refused / failed items."""
    keep = []
    for r in results:
        # Skip if router refused (no contexts retrieved)
        if r.get("route_decision") == "refuse":
            continue
        # Skip if there was an error
        if r.get("error"):
            continue
        # Skip if no contexts
        if not r.get("contexts"):
            continue
        # Skip if empty answer
        if not r.get("answer", "").strip():
            continue
        keep.append(r)
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run ID (folder name). Defaults to latest.")
    args = parser.parse_args()

    run_dir = EVAL_ROOT / args.run if args.run else latest_run_dir()
    raw_file = run_dir / "raw_results.json"
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing: {raw_file}")

    print("=" * 70)
    print("AskAWS Evaluation — Phase 2: RAGAS Scoring")
    print(f"Run: {run_dir.name}")
    print("=" * 70)

    with open(raw_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_results = raw["results"]
    evaluable = filter_evaluable(all_results)
    print(f"Total questions:     {len(all_results)}")
    print(f"Evaluable:           {len(evaluable)}")
    print(f"Skipped (refused/error): {len(all_results) - len(evaluable)}")
    print()

    if not evaluable:
        print("No evaluable items. Exiting.")
        return

    # Build the HuggingFace Dataset RAGAS expects
    ds = Dataset.from_dict({
        "question":      [r["question"] for r in evaluable],
        "answer":        [r["answer"] for r in evaluable],
        "contexts":      [r["contexts"] for r in evaluable],
        "ground_truth":  [r["ground_truth"] for r in evaluable],
    })

    print("Setting up RAGAS with Bedrock Claude + Titan...")
    llm, embeddings = build_ragas_components()

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print(f"Running RAGAS on {len(evaluable)} questions across {len(metrics)} metrics...")
    print("This will take 5-15 minutes (RAGAS makes multiple LLM calls per question).\n")

    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,  # tolerate individual failures
    )

    # Persist scored dataframe as JSON
    df = result.to_pandas()
    scored_file = run_dir / "scored_results.json"
    df.to_json(scored_file, orient="records", indent=2)

    # Aggregate scores
    summary = {
        "run_id": run_dir.name,
        "n_evaluated": len(evaluable),
        "metrics": {},
    }
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            mean = float(df[metric].dropna().mean())
            summary["metrics"][metric] = round(mean, 4)

    # Per-service breakdown
    import math
    services = {}
    for r, row in zip(evaluable, df.to_dict(orient="records")):
        svc = r["service"]
        services.setdefault(svc, []).append(row)
    summary["per_service"] = {}
    for svc, rows in services.items():
        svc_metrics = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            vals = []
            for r in rows:
                v = r.get(metric)
                # Filter out None AND NaN (pandas returns NaN for missing scores)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(v)
            if vals:
                svc_metrics[metric] = round(sum(vals) / len(vals), 4)
            else:
                svc_metrics[metric] = None
        summary["per_service"][svc] = {
            "n": len(rows),
            **svc_metrics,
        }

    summary["scored_at"] = datetime.now().isoformat()

    summary_file = run_dir / "ragas_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Overall ({len(evaluable)} questions):")
    for metric, score in summary["metrics"].items():
        print(f"  {metric:22s} {score:.4f}")
    print()
    print("Per service:")
    for svc, m in summary["per_service"].items():
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "  n/a"
        print(f"  {svc:10s} (n={m['n']})  "
              f"faith={fmt(m.get('faithfulness'))}  "
              f"relev={fmt(m.get('answer_relevancy'))}  "
              f"prec ={fmt(m.get('context_precision'))}  "
              f"recall={fmt(m.get('context_recall'))}")
    print()
    print(f"Detailed scores: {scored_file}")
    print(f"Summary:         {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
