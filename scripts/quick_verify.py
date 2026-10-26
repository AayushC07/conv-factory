#!/usr/bin/env python3
"""Quick verification: 10 conversations to check multi-step improvements."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conversation_factory.pipeline import ConversationPipeline
from conversation_factory.evaluation.metrics import corpus_quality_summary

async def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "stress_test_data"
    output = base / "quick_verify_output"

    if not data_dir.exists():
        print("Run stress_test_50tools_200conv.py first to create test data")
        return

    pipeline = ConversationPipeline(
        data_dir=data_dir, output_dir=output,
        seed=99, cross_conversation_steering=False,
    )
    pipeline.build()

    convs = await pipeline.generate(
        num_conversations=10, min_steps=3, max_steps=6,
        multi_step_ratio=0.80, short_conversation_ratio=0.0,
        min_multi_step_calls=3, concurrency=3,
    )

    all_eps = list(pipeline.graph.graph.nodes())
    cats = pipeline.registry.categories
    metrics = corpus_quality_summary(convs, cats, all_eps)

    print(f"\n{'='*60}")
    print(f"Quick Verify: {len(convs)} conversations")
    print(f"{'='*60}")

    tc_counts = [c.get("metadata", {}).get("num_tool_calls", 0) for c in convs]
    dt_counts = [c.get("metadata", {}).get("num_distinct_tools", 0) for c in convs]

    print(f"  Multi-step ratio (>=3 calls): {metrics['coverage']['multi_step_ratio']:.1%}")
    print(f"  Multi-tool ratio (>=2 tools): {metrics['coverage']['multi_tool_ratio']:.1%}")
    print(f"  Mean tool calls/conv:         {metrics['coverage']['mean_tool_calls']:.2f}")
    print(f"  Mean distinct tools/conv:     {metrics['coverage']['mean_distinct_tools']:.2f}")
    print(f"  Tool call counts:             {tc_counts}")
    print(f"  Distinct tool counts:         {dt_counts}")
    print(f"  Min tool calls:               {min(tc_counts)}")
    print(f"  Mean judge score:             {metrics['quality']['mean_judge_score']:.2f}")

    gen = metrics.get("generation_stats", {})
    if gen:
        print(f"  Failed/attrition:             {gen.get('failed', 0)}/{gen.get('requested', 10)}")

    multi_step_ok = metrics['coverage']['multi_step_ratio'] >= 0.60
    multi_tool_ok = metrics['coverage']['multi_tool_ratio'] >= 0.60
    no_zero = min(tc_counts) >= 1

    print(f"\n  Multi-step >= 60%: {'PASS' if multi_step_ok else 'FAIL'}")
    print(f"  Multi-tool >= 60%: {'PASS' if multi_tool_ok else 'FAIL'}")
    print(f"  No zero-tool convs: {'PASS' if no_zero else 'FAIL'}")

if __name__ == "__main__":
    asyncio.run(main())
