#!/usr/bin/env python3
"""Plot Run A vs Run B diversity & quality comparison from stress test results.

Produces a publication-quality figure showing:
1. Diversity metrics comparison (bar chart)
2. Quality metrics comparison (bar chart)
3. Coverage metrics (multi-step, multi-tool)
4. Category distribution heatmap
"""

import json
import sys
from pathlib import Path

# Try to load results from various sources
base = Path(__file__).resolve().parent.parent


def load_results(
    run_a_path: Path | None = None,
    run_b_path: Path | None = None,
) -> tuple[dict, dict]:
    """Load Run A/B metrics from stress test report, stress dirs, or output/run_{a,b}."""
    if run_a_path is not None and run_b_path is not None:
        with open(run_a_path, encoding="utf-8") as f:
            run_a = json.load(f)
        with open(run_b_path, encoding="utf-8") as f:
            run_b = json.load(f)
        return run_a, run_b

    report_path = base / "stress_test_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("run_a", {}), data.get("run_b", {})

    a_path = base / "stress_output_run_a" / "metrics.json"
    b_path = base / "stress_output_run_b" / "metrics.json"
    if a_path.exists() and b_path.exists():
        with open(a_path, encoding="utf-8") as f:
            run_a = json.load(f)
        with open(b_path, encoding="utf-8") as f:
            run_b = json.load(f)
        return run_a, run_b

    # Default: convfactory generate output dirs (gitignored locally)
    a_path = base / "output" / "run_a" / "metrics.json"
    b_path = base / "output" / "run_b" / "metrics.json"
    if a_path.exists() and b_path.exists():
        with open(a_path, encoding="utf-8") as f:
            run_a = json.load(f)
        with open(b_path, encoding="utf-8") as f:
            run_b = json.load(f)
        return run_a, run_b

    print(
        "No metrics found. Expected one of:\n"
        "  - output/run_a/metrics.json and output/run_b/metrics.json\n"
        "  - stress_output_run_{a,b}/metrics.json\n"
        "  - stress_test_report.json\n"
        "Or pass: python scripts/plot_ab_comparison.py --run-a PATH --run-b PATH",
        file=sys.stderr,
    )
    sys.exit(1)


def load_registry_blurb(run_a_dir: Path) -> tuple[int | None, int | None]:
    """Read num_tools / num_endpoints from build_summary next to a run, if present."""
    summary_path = run_a_dir / "build_summary.json"
    if not summary_path.exists():
        return None, None
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
    reg = data.get("registry", {})
    return reg.get("num_tools"), reg.get("num_endpoints")


def plot(
    run_a: dict,
    run_b: dict,
    *,
    out_path: Path | None = None,
    run_a_dir: Path | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = run_a.get("num_conversations", run_b.get("num_conversations", "?"))
    num_tools, num_endpoints = (None, None)
    if run_a_dir is not None:
        num_tools, num_endpoints = load_registry_blurb(run_a_dir)

    tools_part = (
        f"{num_tools} tools, {num_endpoints} endpoints, "
        if num_tools and num_endpoints
        else ""
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Run A (No Steering) vs Run B (Steering Enabled)\n"
        f"{tools_part}{n} conversations each, seed=42, GPT-4o-mini",
        fontsize=14, fontweight="bold",
    )

    colors_a = "#5C6BC0"  # indigo
    colors_b = "#26A69A"  # teal

    # --- Panel 1: Diversity Metrics ---
    ax = axes[0, 0]
    div_metrics = [
        ("Endpoint\nEntropy (norm)",
         run_a["diversity"]["endpoint_entropy_normalized"],
         run_b["diversity"]["endpoint_entropy_normalized"]),
        ("Category\nCoverage",
         run_a["diversity"]["category_coverage"],
         run_b["diversity"]["category_coverage"]),
        ("Category\nEntropy (norm)",
         run_a["diversity"]["category_entropy"] / run_a["diversity"]["category_entropy_max_bits"],
         run_b["diversity"]["category_entropy"] / run_b["diversity"]["category_entropy_max_bits"]),
        ("Pairwise\nJaccard Dist",
         run_a["diversity"]["mean_pairwise_jaccard_distance"],
         run_b["diversity"]["mean_pairwise_jaccard_distance"]),
    ]
    labels = [m[0] for m in div_metrics]
    vals_a = [m[1] * 100 for m in div_metrics]
    vals_b = [m[2] * 100 for m in div_metrics]

    x = np.arange(len(labels))
    w = 0.35
    bars_a = ax.bar(x - w/2, vals_a, w, label="Run A (no steering)", color=colors_a, alpha=0.85)
    bars_b = ax.bar(x + w/2, vals_b, w, label="Run B (steering)", color=colors_b, alpha=0.85)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Diversity Metrics", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Add value labels
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8, color=colors_a)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8, color=colors_b)

    # --- Panel 2: Quality Metrics ---
    ax = axes[0, 1]
    qual_metrics = [
        ("Mean Judge\nScore",
         run_a["quality"]["mean_judge_score"],
         run_b["quality"]["mean_judge_score"]),
        ("Min Judge\nScore",
         run_a["quality"]["min_judge_score"],
         run_b["quality"]["min_judge_score"]),
        ("Max Judge\nScore",
         run_a["quality"]["max_judge_score"],
         run_b["quality"]["max_judge_score"]),
    ]
    labels = [m[0] for m in qual_metrics]
    vals_a = [m[1] for m in qual_metrics]
    vals_b = [m[2] for m in qual_metrics]

    x = np.arange(len(labels))
    bars_a = ax.bar(x - w/2, vals_a, w, label="Run A", color=colors_a, alpha=0.85)
    bars_b = ax.bar(x + w/2, vals_b, w, label="Run B", color=colors_b, alpha=0.85)

    ax.set_ylabel("Score (1-5)")
    ax.set_title("Quality Metrics (LLM Judge)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.4, label="Threshold (3.0)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color=colors_a)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color=colors_b)

    # --- Panel 3: Coverage Metrics ---
    ax = axes[1, 0]
    cov_metrics = [
        ("Multi-step\n(≥3 calls)",
         run_a["coverage"]["multi_step_ratio"],
         run_b["coverage"]["multi_step_ratio"]),
        ("Multi-tool\n(≥2 tools)",
         run_a["coverage"]["multi_tool_ratio"],
         run_b["coverage"]["multi_tool_ratio"]),
        ("Joint\n(both)",
         run_a["coverage"]["joint_multi_step_and_multi_tool_ratio"],
         run_b["coverage"]["joint_multi_step_and_multi_tool_ratio"]),
    ]
    labels = [m[0] for m in cov_metrics]
    vals_a = [m[1] * 100 for m in cov_metrics]
    vals_b = [m[2] * 100 for m in cov_metrics]

    x = np.arange(len(labels))
    bars_a = ax.bar(x - w/2, vals_a, w, label="Run A", color=colors_a, alpha=0.85)
    bars_b = ax.bar(x + w/2, vals_b, w, label="Run B", color=colors_b, alpha=0.85)

    ax.set_ylabel("Ratio (%)")
    ax.set_title("Coverage: Multi-step & Multi-tool", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.4, label="Target (50%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, color=colors_a)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, color=colors_b)

    # --- Panel 4: Endpoint & Tool Usage ---
    ax = axes[1, 1]
    usage_metrics = [
        ("Mean Tool\nCalls/Conv",
         run_a["coverage"]["mean_tool_calls"],
         run_b["coverage"]["mean_tool_calls"]),
        ("Mean Distinct\nTools/Conv",
         run_a["coverage"]["mean_distinct_tools"],
         run_b["coverage"]["mean_distinct_tools"]),
        ("Mean\nTurns/Conv",
         run_a["coverage"]["mean_turns"] / 5,  # scale down for visual
         run_b["coverage"]["mean_turns"] / 5),
    ]
    labels = [m[0] for m in usage_metrics]
    vals_a = [m[1] for m in usage_metrics]
    vals_b = [m[2] for m in usage_metrics]

    x = np.arange(len(labels))
    bars_a = ax.bar(x - w/2, vals_a, w, label="Run A", color=colors_a, alpha=0.85)
    bars_b = ax.bar(x + w/2, vals_b, w, label="Run B", color=colors_b, alpha=0.85)

    # Show actual values (not scaled) for turns
    actual_a = [run_a["coverage"]["mean_tool_calls"], run_a["coverage"]["mean_distinct_tools"], run_a["coverage"]["mean_turns"]]
    actual_b = [run_b["coverage"]["mean_tool_calls"], run_b["coverage"]["mean_distinct_tools"], run_b["coverage"]["mean_turns"]]

    for i, (bar, val) in enumerate(zip(bars_a, actual_a)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=colors_a)
    for i, (bar, val) in enumerate(zip(bars_b, actual_b)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=colors_b)

    ax.set_ylabel("Count")
    ax.set_title("Usage Statistics", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Add annotation about the turns scaling
    ax.annotate("* Turns divided by 5\n  for visual scale", xy=(0.95, 0.95),
                xycoords="axes fraction", ha="right", va="top", fontsize=7,
                fontstyle="italic", color="gray")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    ent_a = run_a["diversity"]["endpoint_entropy_normalized"]
    ent_b = run_b["diversity"]["endpoint_entropy_normalized"]
    d_ent_pp = (ent_b - ent_a) * 100.0
    jq_a = run_a["quality"]["mean_judge_score"]
    jq_b = run_b["quality"]["mean_judge_score"]
    d_jq = jq_b - jq_a
    ta = run_a.get("time_seconds")
    tb = run_b.get("time_seconds")
    time_part = ""
    if isinstance(ta, (int, float)) and isinstance(tb, (int, float)) and (ta or tb):
        time_part = f"  |  Wall time A: {ta:.0f}s, B: {tb:.0f}s"

    fig.text(
        0.5,
        0.01,
        f"Δ normalized endpoint entropy: {d_ent_pp:+.2f} pp  |  Δ mean judge score: {d_jq:+.3f}{time_part}",
        ha="center",
        fontsize=9,
        fontstyle="italic",
        color="gray",
    )

    dest = out_path if out_path is not None else base / "run_ab_comparison.png"
    plt.savefig(dest, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {dest}")
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot Run A vs Run B from metrics.json files.")
    parser.add_argument(
        "--run-a",
        type=Path,
        default=None,
        help="Path to Run A metrics.json (default: output/run_a/metrics.json)",
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        default=None,
        help="Path to Run B metrics.json (default: output/run_b/metrics.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: run_ab_comparison.png at repo root)",
    )
    args = parser.parse_args()
    run_a, run_b = load_results(args.run_a, args.run_b)
    print(f"Run A: {run_a.get('num_conversations', '?')} convs")
    print(f"Run B: {run_b.get('num_conversations', '?')} convs")

    out = args.output if args.output is not None else base / "run_ab_comparison.png"
    if args.run_a is not None:
        run_a_dir = args.run_a.parent
    else:
        run_a_dir = base / "output" / "run_a"

    plot(run_a, run_b, out_path=out, run_a_dir=run_a_dir if run_a_dir.exists() else None)
