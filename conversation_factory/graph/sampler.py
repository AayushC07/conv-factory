"""Tool-chain sampler with constrained sampling.

Walks the Tool Graph to propose candidate tool chains for conversations.
Supports constraints like:
- Minimum/maximum chain length
- Required categories or tools
- Pattern type (sequential, parallel, mixed)
"""

from __future__ import annotations

import dataclasses
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .tool_graph import ToolGraph

logger = logging.getLogger(__name__)


@dataclass
class SamplingConstraints:
    """Constraints for tool-chain sampling."""

    min_steps: int = 1
    max_steps: int = 5
    required_categories: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    excluded_endpoints: set[str] = field(default_factory=set)
    pattern: str = "sequential"  # sequential | parallel | mixed
    require_multi_tool: bool = False
    min_distinct_tools: int = 1
    #: If True, do not jump to a random node when the walk has no out-neighbors
    #: (avoids breaking plausible ordering; may yield shorter chains / more retries).
    strict_graph_walk: bool = False

    def __post_init__(self) -> None:
        if self.min_steps < 1:
            self.min_steps = 1
        if self.max_steps < self.min_steps:
            self.max_steps = self.min_steps


@dataclass
class ToolChain:
    """A sampled chain of endpoints to use in a conversation."""

    endpoints: list[str]
    pattern: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.endpoints)

    @property
    def distinct_tools(self) -> set[str]:
        return {ep.split("/")[0] for ep in self.endpoints}

    @property
    def categories(self) -> set[str]:
        return set(self.metadata.get("categories", []))

    @property
    def is_multi_tool(self) -> bool:
        return len(self.distinct_tools) >= 2

    @property
    def is_multi_step(self) -> bool:
        return self.num_steps >= 3


class ToolChainSampler:
    """Samples tool chains from the Tool Graph with constraint support."""

    def __init__(self, graph: ToolGraph, seed: int | None = None) -> None:
        self.graph = graph
        self.rng = random.Random(seed)

    def sample(self, constraints: SamplingConstraints | None = None) -> ToolChain:
        """Sample a single tool chain satisfying the given constraints."""
        if constraints is None:
            constraints = SamplingConstraints()

        for attempt in range(50):
            chain = self._attempt_sample(constraints)
            if chain and self._satisfies_constraints(chain, constraints):
                return chain

        # Fallback: relax constraints and build a minimal chain
        logger.warning("Could not satisfy constraints after 50 attempts, using fallback")
        return self._fallback_sample(constraints)

    def sample_batch(
        self,
        n: int,
        constraints: SamplingConstraints | None = None,
        diversity_penalty: dict[str, float] | None = None,
    ) -> list[ToolChain]:
        """Sample n tool chains, optionally penalizing already-used endpoints."""
        chains = []
        usage_counts: dict[str, int] = {}

        for i in range(n):
            effective_constraints = constraints or SamplingConstraints()
            if diversity_penalty:
                # Build exclusion set from overused endpoints
                excluded = {
                    ep for ep, count in usage_counts.items()
                    if count >= diversity_penalty.get("max_reuse", 3)
                }
                effective_constraints = SamplingConstraints(
                    min_steps=effective_constraints.min_steps,
                    max_steps=effective_constraints.max_steps,
                    required_categories=effective_constraints.required_categories,
                    required_tools=effective_constraints.required_tools,
                    excluded_endpoints=effective_constraints.excluded_endpoints | excluded,
                    pattern=effective_constraints.pattern,
                    require_multi_tool=effective_constraints.require_multi_tool,
                    min_distinct_tools=effective_constraints.min_distinct_tools,
                    strict_graph_walk=effective_constraints.strict_graph_walk,
                )

            chain = self.sample(effective_constraints)
            chains.append(chain)

            for ep in chain.endpoints:
                usage_counts[ep] = usage_counts.get(ep, 0) + 1

        return chains

    def _attempt_sample(self, constraints: SamplingConstraints) -> ToolChain | None:
        """Single attempt at sampling a chain."""
        target_len = self.rng.randint(constraints.min_steps, constraints.max_steps)

        # Pick starting node
        start = self._pick_start_node(constraints)
        if not start:
            return None

        # If the start has multiple same-tool data-flow successors, allow ≥3 steps so
        # search→detail→action paths are reachable when max_steps permits.
        same_tool_df_dests = [
            dst
            for _, dst, data in self.graph.graph.out_edges(start, data=True)
            if data.get("data_flow") and dst.split("/")[0] == start.split("/")[0]
        ]
        if len(same_tool_df_dests) >= 2 and target_len < 3:
            target_len = min(constraints.max_steps, max(target_len, 3))

        chain = [start]

        for _ in range(target_len - 1):
            next_node = self._pick_next_node(chain, constraints)
            if next_node is None:
                break
            chain.append(next_node)

        if len(chain) < constraints.min_steps:
            return None

        if constraints.strict_graph_walk and not self._workflow_order_valid(chain):
            return None

        categories = set()
        for ep in chain:
            node_data = self.graph.get_node_data(ep)
            if node_data:
                categories.add(node_data.get("category", "Other"))

        pattern = constraints.pattern
        if pattern == "mixed":
            pattern = self.rng.choice(["sequential", "parallel"])

        return ToolChain(
            endpoints=chain,
            pattern=pattern,
            metadata={"categories": list(categories)},
        )

    def _workflow_order_valid(self, endpoints: list[str]) -> bool:
        """Within each tool, operation ranks must not decrease (no book→search)."""
        by_tool: dict[str, list[int]] = defaultdict(list)
        for ep in endpoints:
            tool = ep.split("/")[0]
            rank = self.graph._get_operation_rank(ep.split("/")[-1])
            by_tool[tool].append(rank)
        for ranks in by_tool.values():
            for a, b in zip(ranks, ranks[1:]):
                if b < a:
                    return False
        return True

    def _pick_start_node(self, constraints: SamplingConstraints) -> str | None:
        """Pick a starting node respecting constraints."""
        candidates = list(self.graph.graph.nodes())

        if constraints.excluded_endpoints:
            candidates = [c for c in candidates if c not in constraints.excluded_endpoints]

        if constraints.required_categories:
            cat = self.rng.choice(constraints.required_categories)
            cat_endpoints = self.graph.get_endpoints_by_category(cat)
            cat_candidates = [c for c in cat_endpoints if c in set(candidates)]
            if cat_candidates:
                candidates = cat_candidates

        if constraints.required_tools:
            tool = self.rng.choice(constraints.required_tools)
            tool_endpoints = self.graph.get_endpoints_by_tool(tool)
            tool_candidates = [c for c in tool_endpoints if c in set(candidates)]
            if tool_candidates:
                candidates = tool_candidates

        if not candidates:
            return None

        # Prefer discovery / list-style endpoints so chains do not start mid-workflow
        search_keywords = (
            "search",
            "list",
            "find",
            "query",
            "get_all",
            "get_forecast",
            "get_current",
        )
        search_candidates = [
            c
            for c in candidates
            if any(k in c.split("/")[-1].lower() for k in search_keywords)
        ]
        if search_candidates:
            return self.rng.choice(search_candidates)

        get_candidates = [
            c
            for c in candidates
            if self.graph.get_node_data(c).get("method") == "GET"
        ]
        if get_candidates:
            return self.rng.choice(get_candidates)

        return self.rng.choice(candidates)

    def _pick_next_node(
        self, chain: list[str], constraints: SamplingConstraints
    ) -> str | None:
        """Pick the next node in the chain using weighted graph traversal."""
        current = chain[-1]
        used = set(chain)

        neighbors = []
        weights = []
        for _, dst, data in self.graph.graph.out_edges(current, data=True):
            if dst in used or dst in constraints.excluded_endpoints:
                continue
            if constraints.strict_graph_walk:
                rel = data.get("relation")
                # Pure category bridges (no parameter/data dependency) are skipped.
                if rel == "same_category" and not data.get("data_flow"):
                    continue
                cur_tool = current.split("/")[0]
                dst_tool = dst.split("/")[0]
                if cur_tool == dst_tool:
                    cr = self.graph._get_operation_rank(current.split("/")[-1])
                    dr = self.graph._get_operation_rank(dst.split("/")[-1])
                    if dr < cr:
                        continue
            neighbors.append(dst)
            w = data.get("weight", 0.5)

            # Boost multi-tool if required
            if constraints.require_multi_tool or constraints.min_distinct_tools > 1:
                current_tools = {ep.split("/")[0] for ep in chain}
                dst_tool = dst.split("/")[0]
                if dst_tool not in current_tools:
                    w *= 2.0

            # Boost required categories
            if constraints.required_categories:
                node_data = self.graph.get_node_data(dst)
                if node_data.get("category") in constraints.required_categories:
                    w *= 1.5

            weights.append(w)

        if not neighbors:
            if constraints.strict_graph_walk:
                return None
            # Try random jump within the same category (can break narrative order)
            node_data = self.graph.get_node_data(current)
            category = node_data.get("category", "Other") if node_data else "Other"
            cat_eps = self.graph.get_endpoints_by_category(category)
            candidates = [ep for ep in cat_eps if ep not in used and ep not in constraints.excluded_endpoints]
            if not candidates:
                all_nodes = list(self.graph.graph.nodes())
                candidates = [n for n in all_nodes if n not in used and n not in constraints.excluded_endpoints]
            if candidates:
                return self.rng.choice(candidates)
            return None

        total = sum(weights)
        if total == 0:
            return self.rng.choice(neighbors)
        probs = [w / total for w in weights]
        return self.rng.choices(neighbors, weights=probs, k=1)[0]

    def _satisfies_constraints(
        self, chain: ToolChain, constraints: SamplingConstraints
    ) -> bool:
        if chain.num_steps < constraints.min_steps:
            return False
        if chain.num_steps > constraints.max_steps:
            return False
        if constraints.require_multi_tool and not chain.is_multi_tool:
            return False
        if len(chain.distinct_tools) < constraints.min_distinct_tools:
            return False
        if constraints.required_categories:
            for cat in constraints.required_categories:
                cat_eps = set(self.graph.get_endpoints_by_category(cat))
                if not any(ep in cat_eps for ep in chain.endpoints):
                    return False
        if constraints.required_tools:
            for tool in constraints.required_tools:
                tool_eps = set(self.graph.get_endpoints_by_tool(tool))
                if not any(ep in tool_eps for ep in chain.endpoints):
                    return False
        if constraints.strict_graph_walk and not self._workflow_order_valid(
            chain.endpoints
        ):
            return False
        return True

    def _fallback_sample(self, constraints: SamplingConstraints) -> ToolChain:
        """Last-resort: retry relaxed walks, then random nodes."""
        relaxed = dataclasses.replace(constraints, strict_graph_walk=False)
        for _ in range(40):
            attempt = self._attempt_sample(relaxed)
            if attempt and self._satisfies_constraints(attempt, constraints):
                attempt.metadata["fallback"] = True
                return attempt

        all_nodes = list(self.graph.graph.nodes())
        available = [n for n in all_nodes if n not in constraints.excluded_endpoints]
        if not available:
            available = all_nodes

        n = min(constraints.min_steps, len(available))
        chosen = self.rng.sample(available, min(n, len(available)))

        categories = set()
        for ep in chosen:
            nd = self.graph.get_node_data(ep)
            if nd:
                categories.add(nd.get("category", "Other"))

        return ToolChain(
            endpoints=chosen,
            pattern=constraints.pattern,
            metadata={"categories": list(categories), "fallback": True},
        )
