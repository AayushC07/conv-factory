"""Cross-conversation steering for corpus-level diversity.

Uses:
- Per-endpoint frequency budgets (caps dominance, not just swapping one hot endpoint
  for another)
- Category-level soft caps (e.g. Travel) alongside endpoint penalties
- Minimum coverage for thin categories (e.g. Weather) so graph neighbours are not
  collateral damage
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

from ..graph.sampler import SamplingConstraints

logger = logging.getLogger(__name__)


class DiversitySteering:
    """Steers generation toward diverse coverage across a corpus."""

    def __init__(
        self,
        enabled: bool = True,
        *,
        max_endpoint_uses: int | None = None,
        travel_category_share_cap: float = 0.46,
        weather_min_fraction_of_corpus: float = 0.12,
    ) -> None:
        self.enabled = enabled
        self._max_endpoint_uses_override = max_endpoint_uses
        self.travel_category_share_cap = travel_category_share_cap
        self.weather_min_fraction_of_corpus = weather_min_fraction_of_corpus

        self._endpoint_counts: Counter[str] = Counter()
        self._category_counts: Counter[str] = Counter()
        self._tool_counts: Counter[str] = Counter()
        self._pattern_counts: Counter[str] = Counter()
        self._pair_counts: Counter[tuple[str, str]] = Counter()
        self._total_conversations = 0
        self._planned_corpus_size: int | None = None

    def set_planned_corpus_size(self, n: int) -> None:
        """Total conversations in this run — used for frequency budgets."""
        self._planned_corpus_size = max(1, n)

    def record_conversation(self, conversation: dict[str, Any]) -> None:
        """Record a generated conversation for diversity tracking."""
        metadata = conversation.get("metadata", {})
        tools_used = metadata.get("tools_used", [])
        categories = metadata.get("categories", [])
        pattern = metadata.get("tool_chain_pattern", "sequential")

        for tool in tools_used:
            self._endpoint_counts[tool] += 1
            tool_name = tool.split("/")[0] if "/" in tool else tool
            self._tool_counts[tool_name] += 1

        for cat in categories:
            self._category_counts[cat] += 1

        self._pattern_counts[pattern] += 1

        for i, t1 in enumerate(sorted(tools_used)):
            for t2 in sorted(tools_used)[i + 1 :]:
                self._pair_counts[(t1, t2)] += 1

        self._total_conversations += 1

    def _budget_max_uses_per_endpoint(self) -> int:
        planned = self._planned_corpus_size or max(self._total_conversations, 1)
        if self._max_endpoint_uses_override is not None:
            return max(2, self._max_endpoint_uses_override)
        # Global frequency budget: cap per-endpoint occurrences.
        # For small corpora (< 15), use a higher floor so we don't starve categories.
        if planned < 15:
            return max(4, int(math.ceil(0.40 * planned)))
        return max(3, int(math.ceil(0.28 * planned)))

    def _excluded_by_frequency_budget(self, all_endpoints: list[str]) -> set[str]:
        cap = self._budget_max_uses_per_endpoint()
        candidates = {ep for ep, c in self._endpoint_counts.items() if c >= cap}
        # Safety: never exclude *all* endpoints of a category — that would reduce
        # category coverage instead of improving diversity.
        if not all_endpoints:
            return candidates
        from collections import defaultdict
        cat_eps: dict[str, list[str]] = defaultdict(list)
        for ep in all_endpoints:
            nd = self._endpoint_counts  # only need category from graph, approximate via node data
            # Group by tool (proxy for category when graph isn't available here)
            tool = ep.split("/")[0] if "/" in ep else ep
            cat_eps[tool].append(ep)
        safe: set[str] = set()
        for tool, eps in cat_eps.items():
            remaining = [e for e in eps if e not in candidates]
            if not remaining and eps:
                # Would eliminate this tool entirely — keep the least-used endpoint
                least = min(eps, key=lambda e: self._endpoint_counts.get(e, 0))
                safe.add(least)
        return candidates - safe

    def _travel_hotspot_exclusions(
        self,
        all_endpoints: list[str],
        endpoint_category: dict[str, str],
    ) -> set[str]:
        """When Travel dominates category tags, thin the heaviest Travel endpoints."""
        n = self._total_conversations
        if n < 8:
            return set()
        travel_tags = self._category_counts.get("Travel", 0)
        if travel_tags / n <= self.travel_category_share_cap:
            return set()
        travel_eps = [
            ep
            for ep in all_endpoints
            if endpoint_category.get(ep) == "Travel"
        ]
        travel_eps.sort(key=lambda e: self._endpoint_counts.get(e, 0), reverse=True)
        # Exclude the two most-used Travel endpoints so Flights/Hotels/maps rotate,
        # not a single replacement hotspot.
        return set(travel_eps[:2])

    def _weather_boost_categories(self, available: list[str]) -> list[str]:
        """Ensure Weather appears often enough (structural neighbour of Travel)."""
        planned = self._planned_corpus_size or max(self._total_conversations, 1)
        min_weather = max(1, int(math.ceil(self.weather_min_fraction_of_corpus * planned)))
        out: list[str] = []
        if (
            "Weather" in available
            and self._category_counts.get("Weather", 0) < min_weather
            and self._total_conversations >= 1
        ):
            out.append("Weather")
        return out

    def get_sampling_constraints(
        self,
        available_categories: list[str],
        base_constraints: SamplingConstraints | None = None,
        *,
        all_endpoints: list[str] | None = None,
        endpoint_category: dict[str, str] | None = None,
    ) -> SamplingConstraints:
        """Apply frequency budgets, category caps, and thin-category boosts."""
        if not self.enabled or self._total_conversations == 0:
            return base_constraints or SamplingConstraints()

        base = base_constraints or SamplingConstraints()
        all_eps = all_endpoints or []
        ecat = endpoint_category or {}

        overused = self._excluded_by_frequency_budget(all_eps) if all_eps else set()
        overused |= self._get_overused_endpoints_legacy()
        if all_eps and ecat:
            overused |= self._travel_hotspot_exclusions(all_eps, ecat)

        required_tools = list(base.required_tools)
        # Only nudge toward unused tools for single-tool conversations;
        # multi-tool conversations already spread across tools via the sampler's
        # cross-tool boost, and forcing a specific tool can make sampling fail.
        if all_eps and ecat and not required_tools and not base.require_multi_tool:
            required_tools = self._unused_tool_nudge(all_eps, ecat) or required_tools

        # Category requirements: Weather floor + deficit (not Travel-only when Travel is hot).
        # When multi-tool is required, forcing a thin category often makes the sampler
        # fail (50 retries → fallback), which *hurts* coverage.  Only force categories
        # for single-tool / unconstrained conversations.
        req_cats: list[str] = list(base.required_categories)
        if not base.require_multi_tool:
            steering_cats: list[str] = []
            steering_cats.extend(self._weather_boost_categories(available_categories))
            deficit = self._get_deficit_categories(available_categories)
            travel_hot = (
                self._total_conversations >= 4
                and self._category_counts.get("Travel", 0) / self._total_conversations
                > self.travel_category_share_cap
            )
            for cat in deficit:
                if travel_hot and cat == "Travel":
                    continue
                if cat not in steering_cats:
                    steering_cats.append(cat)
            if steering_cats:
                req_cats = steering_cats[:3]

        return SamplingConstraints(
            min_steps=base.min_steps,
            max_steps=base.max_steps,
            required_categories=req_cats,
            required_tools=required_tools,
            excluded_endpoints=base.excluded_endpoints | overused,
            pattern=base.pattern,
            require_multi_tool=base.require_multi_tool,
            min_distinct_tools=base.min_distinct_tools,
            strict_graph_walk=base.strict_graph_walk,
        )

    def _unused_tool_nudge(
        self, all_endpoints: list[str], endpoint_category: dict[str, str]
    ) -> list[str] | None:
        """Prefer an unused endpoint whose category is not already overheated."""
        unused = [ep for ep in all_endpoints if self._endpoint_counts.get(ep, 0) == 0]
        if not unused:
            return None
        n = self._total_conversations
        travel_hot = (
            n >= 3
            and self._category_counts.get("Travel", 0) / max(n, 1) > 0.4
        )
        for ep in sorted(unused):
            cat = endpoint_category.get(ep, "Other")
            if travel_hot and cat == "Travel":
                continue
            return [ep.split("/")[0]]
        return [sorted(unused)[0].split("/")[0]]

    def get_diversity_penalty(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        avg_usage = sum(self._endpoint_counts.values()) / max(
            len(self._endpoint_counts), 1
        )
        return {"max_reuse": max(int(avg_usage * 2), 3)}

    def _get_deficit_categories(self, available: list[str]) -> list[str]:
        if not available or self._total_conversations < 2:
            return []
        expected = self._total_conversations / max(len(available), 1)
        deficits: list[tuple[str, float]] = []
        for cat in available:
            actual = self._category_counts.get(cat, 0)
            if actual < expected * 0.5:
                deficits.append((cat, expected - actual))
        deficits.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in deficits]

    def _get_overused_endpoints_legacy(self) -> set[str]:
        """Legacy dominance rule — combined with hard frequency budget."""
        if not self._endpoint_counts or self._total_conversations < 3:
            return set()
        total_usage = sum(self._endpoint_counts.values())
        n_unique = len(self._endpoint_counts)
        fair_share = total_usage / max(n_unique, 1)
        overused: set[str] = set()
        for ep, count in self._endpoint_counts.items():
            if count >= 5 and (count > fair_share * 3 or count > total_usage * 0.5):
                overused.add(ep)
        return overused

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_conversations": self._total_conversations,
            "unique_endpoints_used": len(self._endpoint_counts),
            "unique_categories_used": len(self._category_counts),
            "unique_tools_used": len(self._tool_counts),
            "top_endpoints": self._endpoint_counts.most_common(10),
            "top_categories": self._category_counts.most_common(10),
            "pattern_distribution": dict(self._pattern_counts),
            "max_endpoint_uses_budget": (
                self._budget_max_uses_per_endpoint() if self._endpoint_counts else None
            ),
        }

    def reset(self) -> None:
        self._endpoint_counts.clear()
        self._category_counts.clear()
        self._tool_counts.clear()
        self._pattern_counts.clear()
        self._pair_counts.clear()
        self._total_conversations = 0
        self._planned_corpus_size = None
