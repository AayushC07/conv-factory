#!/usr/bin/env python3
"""Scaling benchmark: measure performance across tool counts.

Runs 100 conversations at each scale point (5, 10, 15, 20, 25, 30, 35, 40 tools)
and plots how multi-step ratio, multi-tool ratio, judge score, and diversity
metrics change as the tool registry grows.
"""

import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conversation_factory.pipeline import ConversationPipeline
from conversation_factory.evaluation.metrics import corpus_quality_summary

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Import tool definitions from the stress test
from stress_test_50tools_200conv import TOOL_TEMPLATES, CATEGORIES, build_tool_json

# ---------------------------------------------------------------------------
# Scale points
# ---------------------------------------------------------------------------
SCALE_POINTS = [5, 10, 15, 20, 25, 30, 35, 40]
CONVS_PER_POINT = 100
CONCURRENCY = 3  # respect rate limits


def create_dataset_with_n_tools(n_tools: int, output_dir: Path) -> int:
    """Write exactly n_tools from the template pool in category layout."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Flatten all tools in category order
    all_tools = []
    for category in CATEGORIES:
        for tool_name, description, endpoints in TOOL_TEMPLATES.get(category, []):
            all_tools.append((category, tool_name, description, endpoints))

    # Take first n_tools (spread across categories naturally)
    selected = all_tools[:n_tools]
    count = 0
    for category, tool_name, description, endpoints in selected:
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        tool_json = build_tool_json(tool_name, description, endpoints, category)
        with open(cat_dir / f"{tool_name.lower()}.json", "w") as f:
            json.dump(tool_json, f, indent=2)
        count += 1
    return count


async def run_single_scale_point(n_tools: int, base_dir: Path) -> dict:
    """Run 100 conversations with n_tools and return metrics."""
    data_dir = base_dir / f"scale_data_{n_tools}"
    output_dir = base_dir / f"scale_output_{n_tools}"

    actual = create_dataset_with_n_tools(n_tools, data_dir)
    print(f"\n  [{n_tools} tools] Created {actual} tools, generating {CONVS_PER_POINT} conversations...")

    pipeline = ConversationPipeline(
        data_dir=data_dir, output_dir=output_dir,
        seed=42, cross_conversation_steering=True,
    )
    build_info = pipeline.build()
    num_endpoints = build_info["registry"]["num_endpoints"]
    print(f"  [{n_tools} tools] Registry: {actual} tools, {num_endpoints} endpoints")

    t0 = time.time()
    convs = await pipeline.generate(
        num_conversations=CONVS_PER_POINT,
        min_steps=1, max_steps=8,
        multi_step_ratio=0.65,
        short_conversation_ratio=0.10,
        min_multi_step_calls=3,
        concurrency=CONCURRENCY,
    )
    elapsed = time.time() - t0

    all_eps = list(pipeline.graph.graph.nodes())
    cats = pipeline.registry.categories
    metrics = corpus_quality_summary(convs, cats, all_eps)

    # Extra stats
    tc_list = [c.get("metadata", {}).get("num_tool_calls", 0) for c in convs]
    dt_list = [c.get("metadata", {}).get("num_distinct_tools", 0) for c in convs]

    result = {
        "n_tools": n_tools,
        "n_endpoints": num_endpoints,
        "n_conversations": len(convs),
        "elapsed_seconds": round(elapsed, 1),
        "multi_step_ratio": metrics["coverage"]["multi_step_ratio"],
        "multi_tool_ratio": metrics["coverage"]["multi_tool_ratio"],
        "joint_ratio": metrics["coverage"]["joint_multi_step_and_multi_tool_ratio"],
        "mean_tool_calls": metrics["coverage"]["mean_tool_calls"],
        "mean_distinct_tools": metrics["coverage"]["mean_distinct_tools"],
        "mean_judge_score": metrics["quality"]["mean_judge_score"],
        "endpoint_entropy_norm": metrics["diversity"].get("endpoint_entropy_normalized", 0),
        "category_coverage": metrics["diversity"]["category_coverage"],
        "mean_jaccard": metrics["diversity"]["mean_pairwise_jaccard_distance"],
        "min_tool_calls": min(tc_list) if tc_list else 0,
        "max_tool_calls": max(tc_list) if tc_list else 0,
        "zero_tool_convs": sum(1 for t in tc_list if t == 0),
    }

    print(f"  [{n_tools} tools] Done in {elapsed:.0f}s | "
          f"multi-step={result['multi_step_ratio']:.1%} | "
          f"multi-tool={result['multi_tool_ratio']:.1%} | "
          f"judge={result['mean_judge_score']:.2f} | "
          f"entropy={result['endpoint_entropy_norm']:.1%}")

    # Cleanup data dir to save space
    shutil.rmtree(data_dir, ignore_errors=True)

    return result


def plot_results(results: list[dict], output_path: Path):
    """Generate scaling benchmark plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    tools = [r["n_tools"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Scaling Benchmark: Performance vs Tool Count (100 convs each)", fontsize=16, fontweight="bold")

    # --- Plot 1: Multi-step & Multi-tool ratios ---
    ax = axes[0, 0]
    ax.plot(tools, [r["multi_step_ratio"] * 100 for r in results], "o-", color="#2196F3", linewidth=2, label="Multi-step (>=3 calls)")
    ax.plot(tools, [r["multi_tool_ratio"] * 100 for r in results], "s-", color="#4CAF50", linewidth=2, label="Multi-tool (>=2 tools)")
    ax.plot(tools, [r["joint_ratio"] * 100 for r in results], "^-", color="#FF9800", linewidth=2, label="Joint (both)")
    ax.axhline(y=65, color="red", linestyle="--", alpha=0.5, label="Target (65%)")
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Ratio (%)")
    ax.set_title("Multi-step & Multi-tool Ratios")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(tools)

    # --- Plot 2: Mean tool calls & distinct tools ---
    ax = axes[0, 1]
    ax.plot(tools, [r["mean_tool_calls"] for r in results], "o-", color="#9C27B0", linewidth=2, label="Mean tool calls")
    ax.plot(tools, [r["mean_distinct_tools"] for r in results], "s-", color="#00BCD4", linewidth=2, label="Mean distinct tools")
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Count")
    ax.set_title("Tool Usage per Conversation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(tools)

    # --- Plot 3: Judge score ---
    ax = axes[0, 2]
    ax.plot(tools, [r["mean_judge_score"] for r in results], "o-", color="#F44336", linewidth=2, markersize=8)
    ax.axhline(y=3.0, color="orange", linestyle="--", alpha=0.5, label="Min threshold (3.0)")
    ax.fill_between(tools, 3.0, [r["mean_judge_score"] for r in results], alpha=0.15, color="#F44336")
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Score (1-5)")
    ax.set_title("Mean Judge Score")
    ax.legend(fontsize=9)
    ax.set_ylim(1, 5.5)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(tools)

    # --- Plot 4: Endpoint entropy ---
    ax = axes[1, 0]
    ax.plot(tools, [r["endpoint_entropy_norm"] * 100 for r in results], "o-", color="#795548", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Normalized Entropy (%)")
    ax.set_title("Endpoint Entropy (Diversity)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(tools)

    # --- Plot 5: Category coverage & Jaccard ---
    ax = axes[1, 1]
    ax.plot(tools, [r["category_coverage"] * 100 for r in results], "o-", color="#E91E63", linewidth=2, label="Category coverage")
    ax.plot(tools, [r["mean_jaccard"] * 100 for r in results], "s-", color="#3F51B5", linewidth=2, label="Mean pairwise Jaccard")
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Category Coverage & Diversity")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(tools)

    # --- Plot 6: Time per scale point ---
    ax = axes[1, 2]
    bars = ax.bar(tools, [r["elapsed_seconds"] for r in results], color="#607D8B", width=3, alpha=0.8)
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Seconds")
    ax.set_title("Generation Time (100 convs)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(tools)
    # Add value labels on bars
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{r['elapsed_seconds']:.0f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def print_summary_table(results: list[dict]):
    """Print a formatted summary table."""
    print(f"\n{'='*110}")
    print(f"{'Tools':>6} {'EPs':>5} {'Convs':>6} {'Multi-step':>11} {'Multi-tool':>11} {'Joint':>8} "
          f"{'MeanTC':>7} {'Judge':>7} {'Entropy':>8} {'CatCov':>8} {'Jaccard':>8} {'Time':>7}")
    print(f"{'-'*110}")
    for r in results:
        print(f"{r['n_tools']:>6} {r['n_endpoints']:>5} {r['n_conversations']:>6} "
              f"{r['multi_step_ratio']:>10.1%} {r['multi_tool_ratio']:>10.1%} {r['joint_ratio']:>7.1%} "
              f"{r['mean_tool_calls']:>7.2f} {r['mean_judge_score']:>7.2f} "
              f"{r['endpoint_entropy_norm']:>7.1%} {r['category_coverage']:>7.1%} "
              f"{r['mean_jaccard']:>7.1%} {r['elapsed_seconds']:>6.0f}s")
    print(f"{'='*110}")


async def main():
    base_dir = Path(__file__).resolve().parent.parent / "scaling_benchmark"
    base_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("SCALING BENCHMARK")
    print(f"Scale points: {SCALE_POINTS} tools")
    print(f"Conversations per point: {CONVS_PER_POINT}")
    print(f"Concurrency: {CONCURRENCY}")
    print("=" * 70)

    results = []
    total_start = time.time()

    for n_tools in SCALE_POINTS:
        result = await run_single_scale_point(n_tools, base_dir)
        results.append(result)

        # Save intermediate results after each point
        with open(base_dir / "scaling_results.json", "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - total_start

    # Summary
    print_summary_table(results)
    print(f"\nTotal benchmark time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Plot
    plot_path = base_dir / "scaling_benchmark.png"
    plot_results(results, plot_path)

    # Save final results
    with open(base_dir / "scaling_results.json", "w") as f:
        json.dump({"results": results, "total_time_seconds": round(total_time, 1)}, f, indent=2)

    print(f"\nAll results saved to: {base_dir}")


if __name__ == "__main__":
    asyncio.run(main())
