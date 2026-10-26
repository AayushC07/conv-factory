#!/usr/bin/env python3
"""Print a concise review of a generated dataset + evaluation/metrics JSON.

Usage:
  python scripts/review_corpus.py --output-dir output
  python scripts/review_corpus.py --output-dir output/run_b --dataset output/run_b/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Review conversation-factory corpus outputs")
    p.add_argument("--output-dir", "-o", default="output", type=Path, help="Directory with artifacts")
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset.jsonl (default: OUTPUT_DIR/dataset.jsonl)",
    )
    args = p.parse_args()
    out = args.output_dir
    dataset_path = args.dataset or (out / "dataset.jsonl")
    eval_path = out / "evaluation.json"
    metrics_path = out / "metrics.json"

    if not dataset_path.exists():
        raise SystemExit(f"Missing dataset: {dataset_path}")

    convs = _load_jsonl(dataset_path)
    n = len(convs)

    means: list[float] = []
    mins: list[float] = []
    repair_ok = 0
    repair_fail = 0
    for c in convs:
        js = c.get("judge_scores") or {}
        if js:
            vals = list(js.values())
            means.append(sum(vals) / len(vals))
            mins.append(min(vals))
        ra = (c.get("metadata") or {}).get("repair_action")
        if ra == "repaired":
            repair_ok += 1
        elif isinstance(ra, str) and "failed" in ra:
            repair_fail += 1

    print("=== Corpus review ===")
    print(f"Dataset: {dataset_path}")
    print(f"Conversations: {n}")
    if means:
        print(
            f"Judge mean (per-conversation avg of dims): "
            f"{statistics.mean(means):.2f}  (min of those: {min(means):.2f}, max: {max(means):.2f})"
        )
        print(f"Worst single dimension (per conv min): min={min(mins):.2f}, median={statistics.median(mins):.2f}")
    if repair_ok or repair_fail:
        print(f"Repair: repaired={repair_ok}, failed_validation≈{repair_fail} (see metadata.repair_action)")

    for label, path in ("evaluation.json", eval_path), ("metrics.json", metrics_path):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"\n--- {label} ---")
            q = data.get("quality") or {}
            cov = data.get("coverage") or {}
            div = data.get("diversity") or {}
            if q:
                print(f"  mean_judge_score: {q.get('mean_judge_score')}")
                print(f"  min/max judge: {q.get('min_judge_score')} / {q.get('max_judge_score')}")
            if cov:
                print(
                    f"  multi_step_ratio: {cov.get('multi_step_ratio')}  "
                    f"multi_tool_ratio: {cov.get('multi_tool_ratio')}  "
                    f"joint: {cov.get('joint_multi_step_and_multi_tool_ratio')}"
                )
                print(
                    f"  mean_tool_calls: {cov.get('mean_tool_calls')}  "
                    f"mean_turns: {cov.get('mean_turns')}"
                )
            if div:
                print(f"  category_coverage: {div.get('category_coverage')}")
                print(f"  endpoint_entropy_normalized: {div.get('endpoint_entropy_normalized')}")

    print("\nTip: open output/dataset_viewer.html in a browser if present for transcript review.")


if __name__ == "__main__":
    main()
