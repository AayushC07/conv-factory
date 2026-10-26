"""Diversity and quality metrics for corpus-level evaluation.

Run A vs Run B:
- Endpoint entropy (raw) and endpoint_entropy_normalized (÷ log2(|graph|))
- Category entropy over category mentions (captures skew; coverage alone can saturate)
- Category coverage, mean pairwise Jaccard distance
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def endpoint_entropy(conversations: list[dict[str, Any]]) -> float:
    """Shannon entropy of endpoint usage across all conversations (support = used endpoints).

    Higher entropy = more uniform distribution = more diverse tool usage.
    """
    endpoint_counts: Counter[str] = Counter()
    for conv in conversations:
        for tool in conv.get("metadata", {}).get("tools_used", []):
            endpoint_counts[tool] += 1

    total = sum(endpoint_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in endpoint_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def endpoint_entropy_normalized(
    conversations: list[dict[str, Any]], all_endpoints: list[str]
) -> float:
    """Entropy of endpoint usage divided by log2(N) for N = len(all_endpoints).

    Uses the same mass distribution as ``endpoint_entropy`` (only observed endpoints
    get positive probability); the normalization constant is fixed by the full
    graph so Run A/B comparisons use a common scale.
    """
    if not all_endpoints:
        return 0.0
    h = endpoint_entropy(conversations)
    denom = math.log2(len(all_endpoints))
    return h / denom if denom > 0 else 0.0


def category_entropy(
    conversations: list[dict[str, Any]], all_categories: list[str]
) -> float:
    """Shannon entropy of category labels over all conversation-category mentions.

    Each conversation contributes one count per category in ``metadata.categories``.
    """
    if not all_categories or not conversations:
        return 0.0

    counts: Counter[str] = Counter()
    for conv in conversations:
        for cat in conv.get("metadata", {}).get("categories", []):
            counts[cat] += 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for cat in all_categories:
        c = counts.get(cat, 0)
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)

    return entropy


def category_coverage(
    conversations: list[dict[str, Any]], all_categories: list[str]
) -> float:
    """Fraction of available categories represented in the corpus."""
    if not all_categories:
        return 0.0

    used_categories: set[str] = set()
    for conv in conversations:
        cats = conv.get("metadata", {}).get("categories", [])
        used_categories.update(cats)

    return len(used_categories) / len(all_categories)


def mean_pairwise_jaccard_distance(
    conversations: list[dict[str, Any]],
    max_sample_pairs: int = 50_000,
) -> float:
    """Mean Jaccard distance between tool sets across conversation pairs.

    Jaccard distance = 1 - |A ∩ B| / |A ∪ B|
    Higher = more diverse (less overlap between conversations).

    For large corpora (>500 conversations), uses random sampling to avoid
    O(n²) blowup while maintaining statistical accuracy.
    """
    import random as _random

    tool_sets = []
    for conv in conversations:
        tools = set(conv.get("metadata", {}).get("tools_used", []))
        if tools:
            tool_sets.append(tools)

    if len(tool_sets) < 2:
        return 0.0

    n = len(tool_sets)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_sample_pairs:
        # Exact computation for small corpora
        total_distance = 0.0
        num_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                intersection = len(tool_sets[i] & tool_sets[j])
                union = len(tool_sets[i] | tool_sets[j])
                if union > 0:
                    total_distance += 1.0 - (intersection / union)
                    num_pairs += 1
        return total_distance / num_pairs if num_pairs > 0 else 0.0
    else:
        # Sampling-based approximation for large corpora
        rng = _random.Random(42)
        total_distance = 0.0
        num_pairs = 0
        for _ in range(max_sample_pairs):
            i, j = rng.sample(range(n), 2)
            intersection = len(tool_sets[i] & tool_sets[j])
            union = len(tool_sets[i] | tool_sets[j])
            if union > 0:
                total_distance += 1.0 - (intersection / union)
                num_pairs += 1
        return total_distance / num_pairs if num_pairs > 0 else 0.0


def corpus_quality_summary(
    conversations: list[dict[str, Any]],
    all_categories: list[str] | None = None,
    all_endpoints: list[str] | None = None,
) -> dict[str, Any]:
    """Compute all diversity and quality metrics for a corpus.

    ``all_endpoints`` should be the full list of graph endpoint FQNs so
    ``endpoint_entropy_normalized`` uses a stable denominator.
    """
    # Quality metrics from judge scores
    scores = []
    for conv in conversations:
        judge = conv.get("judge_scores", {})
        if judge:
            mean = sum(judge.values()) / len(judge) if judge else 0
            scores.append(mean)

    # Conversation stats
    tool_counts = [conv.get("metadata", {}).get("num_tool_calls", 0) for conv in conversations]
    distinct_tools = [conv.get("metadata", {}).get("num_distinct_tools", 0) for conv in conversations]
    turn_counts = [conv.get("metadata", {}).get("num_turns", 0) for conv in conversations]

    multi_step = sum(1 for tc in tool_counts if tc >= 3)
    multi_tool = sum(1 for dt in distinct_tools if dt >= 2)
    joint_multi = sum(
        1
        for tc, dt in zip(tool_counts, distinct_tools)
        if tc >= 3 and dt >= 2
    )

    cats = all_categories or []
    eps = all_endpoints or []

    diversity: dict[str, Any] = {
        "endpoint_entropy": endpoint_entropy(conversations),
        "category_coverage": category_coverage(conversations, cats),
        "category_entropy": category_entropy(conversations, cats),
        "mean_pairwise_jaccard_distance": mean_pairwise_jaccard_distance(conversations),
    }
    if eps:
        diversity["endpoint_entropy_normalized"] = endpoint_entropy_normalized(
            conversations, eps
        )
        diversity["endpoint_entropy_max_bits"] = math.log2(len(eps)) if eps else 0.0
    if cats:
        diversity["category_entropy_max_bits"] = math.log2(len(cats))

    result = {
        "num_conversations": len(conversations),
        "diversity": diversity,
        "quality": {
            "mean_judge_score": sum(scores) / len(scores) if scores else 0.0,
            "min_judge_score": min(scores) if scores else 0.0,
            "max_judge_score": max(scores) if scores else 0.0,
        },
        "coverage": {
            "multi_step_ratio": multi_step / len(conversations) if conversations else 0.0,
            "multi_tool_ratio": multi_tool / len(conversations) if conversations else 0.0,
            "joint_multi_step_and_multi_tool_ratio": joint_multi / len(conversations)
            if conversations
            else 0.0,
            "mean_tool_calls": sum(tool_counts) / len(tool_counts) if tool_counts else 0.0,
            "mean_distinct_tools": sum(distinct_tools) / len(distinct_tools) if distinct_tools else 0.0,
            "mean_turns": sum(turn_counts) / len(turn_counts) if turn_counts else 0.0,
        },
    }

    return result
