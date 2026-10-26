"""Tool Graph construction.

Builds a directed graph where:
- Nodes are endpoints (fully-qualified: tool_name/endpoint_name)
- Edges represent potential chaining relationships:
  1. Same-tool edges: weighted by REST-style operation rank + declaration tiebreak
  2. Category edges: endpoints across tools in the same category (medium weight)
  3. Parameter-compatibility edges: output of one endpoint can feed input of another
     (highest weight — these represent real data-flow dependencies)
  4. Shared-parameter edges: cross-tool links when inputs share meaningful params (city, date, …)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from ..registry.models import Endpoint, ParamType, Tool
from ..registry.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Param types considered compatible for data-flow edges
_COMPATIBLE_TYPES = {
    ParamType.STRING: {ParamType.STRING},
    ParamType.INTEGER: {ParamType.INTEGER, ParamType.NUMBER},
    ParamType.NUMBER: {ParamType.NUMBER, ParamType.INTEGER},
    ParamType.BOOLEAN: {ParamType.BOOLEAN},
}

# Common ID-like parameter name patterns that signal chaining potential
_ID_PATTERNS = {"id", "ids", "identifier", "key", "code", "number", "ref"}

# Substrings in endpoint names → operation phase (1 = discovery … 5 = destructive).
# More specific keys are matched first (see _get_operation_rank).
_OPERATION_KEYWORDS: list[tuple[str, int]] = [
    ("get_forecast", 1),
    ("get_current", 1),
    ("search", 1),
    ("list", 1),
    ("find", 1),
    ("query", 1),
    ("get_all", 1),
    ("get", 2),
    ("detail", 2),
    ("info", 2),
    ("fetch", 2),
    ("create", 3),
    ("book", 3),
    ("reserve", 3),
    ("make", 3),
    ("purchase", 3),
    ("submit", 3),
    ("confirm", 3),
    ("update", 4),
    ("modify", 4),
    ("edit", 4),
    ("delete", 5),
    ("cancel", 5),
    ("remove", 5),
]

# Cross-domain bridges: shared input parameter names (lowercase)
_SHARED_PARAM_NAMES = frozenset(
    {
        "city",
        "location",
        "date",
        "origin",
        "destination",
        "currency",
        "country",
        "language",
        "query",
    }
)


def _is_id_like(name: str) -> bool:
    lower = name.lower()
    parts = lower.replace("-", "_").split("_")
    return any(p in _ID_PATTERNS for p in parts) or lower.endswith("_id")


class ToolGraph:
    """A directed graph capturing semantic and data-flow relationships between endpoints."""

    @staticmethod
    def _get_operation_rank(endpoint_name: str) -> int:
        """Map endpoint name to a coarse operation phase (discovery → action → delete)."""
        name = endpoint_name.lower()
        for keyword, rank in _OPERATION_KEYWORDS:
            if keyword in name:
                return rank
        return 2

    def _data_flow_weight_by_rank(self, src_fqn: str, dst_fqn: str) -> float:
        """Weight for a data-flow edge: prefer adjacent operation phases over skipping ranks."""
        src_rank = self._get_operation_rank(src_fqn.split("/")[-1])
        dst_rank = self._get_operation_rank(dst_fqn.split("/")[-1])
        if dst_rank > src_rank:
            gap = dst_rank - src_rank
            return 1.0 if gap == 1 else 0.3
        if dst_rank == src_rank:
            return 1.0
        # Backward in coarse rank — keep possible but weaker than skip-forward
        return 0.55

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._tool_nodes: dict[str, list[str]] = defaultdict(list)
        self._category_nodes: dict[str, list[str]] = defaultdict(list)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def build(self, registry: ToolRegistry) -> None:
        """Construct the graph from a tool registry."""
        logger.info("Building tool graph...")

        self._add_nodes(registry)
        self._add_same_tool_edges(registry)
        self._add_category_edges(registry)
        self._add_parameter_compatibility_edges(registry)
        self._add_shared_param_edges(registry)

        logger.info(
            f"Tool graph built: {self.num_nodes} nodes, {self.num_edges} edges"
        )

    def _add_nodes(self, registry: ToolRegistry) -> None:
        for tool in registry.tools:
            for order, ep in enumerate(tool.endpoints):
                fqn = f"{tool.tool_name}/{ep.name}"
                self.graph.add_node(
                    fqn,
                    tool_name=tool.tool_name,
                    endpoint_name=ep.name,
                    category=tool.category.name,
                    description=ep.description,
                    method=ep.method,
                    num_params=len(ep.parameters),
                    required_params=[p.name for p in ep.required_parameters],
                    param_names=[p.name for p in ep.parameters],
                    endpoint_index=order,
                )
                self._tool_nodes[tool.tool_name].append(fqn)
                self._category_nodes[tool.category.name].append(fqn)

    def _add_same_tool_edges(self, registry: ToolRegistry) -> None:
        """Same-tool edges: keyword-based operation rank (primary), declaration index (tiebreak).

        Forward discovery→action edges get high weight; backward edges get a small weight
        so the sampler rarely walks book→search but the graph stays connected.
        """
        for _tool_name, endpoints in self._tool_nodes.items():
            for src in endpoints:
                for dst in endpoints:
                    if src == dst:
                        continue
                    if self.graph.has_edge(src, dst) and self.graph.edges[src, dst].get(
                        "data_flow"
                    ):
                        continue
                    src_rank = self._get_operation_rank(src.split("/")[-1])
                    dst_rank = self._get_operation_rank(dst.split("/")[-1])
                    src_idx = self.graph.nodes[src].get("endpoint_index", 0)
                    dst_idx = self.graph.nodes[dst].get("endpoint_index", 0)
                    if src_rank < dst_rank:
                        gap = dst_rank - src_rank
                        # Prefer adjacent phases (search→get) over skipping (search→book)
                        weight = 0.8 if gap == 1 else 0.3
                        extra = {
                            "workflow_forward": True,
                            "workflow_backward": False,
                            "data_flow": False,
                        }
                    elif src_rank > dst_rank:
                        weight = 0.05
                        # Backward within tool (e.g. book→search): must not be used as a strict walk step
                        extra = {
                            "workflow_forward": False,
                            "workflow_backward": True,
                            "data_flow": False,
                        }
                    else:
                        weight = 0.5 if src_idx < dst_idx else 0.2
                        extra = {
                            "workflow_forward": False,
                            "workflow_backward": False,
                            "data_flow": False,
                        }
                    self.graph.add_edge(
                        src, dst, relation="same_tool", weight=weight, **extra
                    )

    @staticmethod
    def _category_cross_tool_weight(endpoint_fqn: str) -> float:
        """Prefer cross-category jumps onto discovery endpoints (search/list) over book/action."""
        r = ToolGraph._get_operation_rank(endpoint_fqn.split("/")[-1])
        if r == 1:
            return 0.4
        if r == 2:
            return 0.2
        return 0.1

    def _add_category_edges(self, registry: ToolRegistry) -> None:
        """Cross-tool links within one ToolBench category.

        Weights favor landing on search/discovery endpoints (rank 1) over get/detail (2)
        or book/action (3+), so jumps from e.g. Flights→Hotels tend to hit search_hotels first.
        """
        for category, endpoints in self._category_nodes.items():
            tool_groups: dict[str, list[str]] = defaultdict(list)
            for fqn in endpoints:
                tool_name = fqn.split("/")[0]
                tool_groups[tool_name].append(fqn)

            tool_names = list(tool_groups.keys())
            for i, t1 in enumerate(tool_names):
                for t2 in tool_names[i + 1:]:
                    for src in tool_groups[t1]:
                        for dst in tool_groups[t2]:
                            if not self.graph.has_edge(src, dst):
                                w = self._category_cross_tool_weight(dst)
                                self.graph.add_edge(
                                    src,
                                    dst,
                                    relation="same_category",
                                    weight=w,
                                    data_flow=False,
                                )
                            if not self.graph.has_edge(dst, src):
                                w_rev = self._category_cross_tool_weight(src)
                                self.graph.add_edge(
                                    dst,
                                    src,
                                    relation="same_category",
                                    weight=w_rev,
                                    data_flow=False,
                                )

    def _add_parameter_compatibility_edges(self, registry: ToolRegistry) -> None:
        """If endpoint A's output could feed endpoint B's input, add a data-flow edge."""
        endpoints_info: list[tuple[str, Endpoint]] = []
        for tool in registry.tools:
            for ep in tool.endpoints:
                fqn = f"{tool.tool_name}/{ep.name}"
                endpoints_info.append((fqn, ep))

        output_fields: dict[str, list[tuple[str, ParamType]]] = {}
        for fqn, ep in endpoints_info:
            fields = []
            for rf in ep.response_schema:
                fields.append((rf.name, rf.field_type))
            # Heuristic: GET/search endpoints likely return IDs
            if ep.method in ("GET",) and not fields:
                fields.append(("id", ParamType.STRING))
            output_fields[fqn] = fields

        for src_fqn, src_outputs in output_fields.items():
            for dst_fqn, dst_ep in endpoints_info:
                if src_fqn == dst_fqn:
                    continue
                for out_name, out_type in src_outputs:
                    for param in dst_ep.parameters:
                        if self._params_compatible(
                            out_name,
                            out_type,
                            param.name,
                            param.param_type,
                            source_fqn=src_fqn,
                        ):
                            df_w = self._data_flow_weight_by_rank(src_fqn, dst_fqn)
                            if self.graph.has_edge(src_fqn, dst_fqn):
                                # Upgrade to a real data-flow edge (takes precedence over
                                # same_tool / same_category bridges).
                                existing = self.graph.edges[src_fqn, dst_fqn]
                                existing["weight"] = max(existing.get("weight", 0), df_w)
                                existing["data_flow"] = True
                                existing["relation"] = "data_flow"
                                existing["source_field"] = out_name
                                existing["target_param"] = param.name
                            else:
                                self.graph.add_edge(
                                    src_fqn,
                                    dst_fqn,
                                    relation="data_flow",
                                    weight=df_w,
                                    source_field=out_name,
                                    target_param=param.name,
                                )

    def _add_shared_param_edges(self, registry: ToolRegistry) -> None:
        """Cross-tool edges when endpoints share meaningful input parameters (city, date, …)."""
        endpoints_info: list[tuple[str, set[str], str]] = []
        for tool in registry.tools:
            for ep in tool.endpoints:
                fqn = f"{tool.tool_name}/{ep.name}"
                param_names = {p.name.lower() for p in ep.parameters}
                endpoints_info.append((fqn, param_names, tool.tool_name))

        n = len(endpoints_info)
        for i in range(n):
            src_fqn, src_params, src_tool = endpoints_info[i]
            for j in range(i + 1, n):
                dst_fqn, dst_params, dst_tool = endpoints_info[j]
                if src_tool == dst_tool:
                    continue
                shared = (src_params & dst_params) & _SHARED_PARAM_NAMES
                if not shared:
                    continue
                w = min(0.3, 0.1 * len(shared))
                self._merge_or_add_shared_param_edge(src_fqn, dst_fqn, shared, w)
                self._merge_or_add_shared_param_edge(dst_fqn, src_fqn, shared, w)

    def _merge_or_add_shared_param_edge(
        self,
        u: str,
        v: str,
        shared: set[str],
        base_weight: float,
    ) -> None:
        shared_list = sorted(shared)
        if self.graph.has_edge(u, v):
            existing = self.graph.edges[u, v]
            existing["shared_params"] = shared_list
            existing["weight"] = min(1.0, existing.get("weight", 0) + 0.1)
        else:
            self.graph.add_edge(
                u,
                v,
                relation="shared_params",
                weight=base_weight,
                shared_params=shared_list,
            )

    @staticmethod
    def _params_compatible(
        out_name: str,
        out_type: ParamType,
        in_name: str,
        in_type: ParamType,
        *,
        source_fqn: str = "",
    ) -> bool:
        """Heuristic: compatible types and names; generic *id* output is entity-aware."""
        out_lower = out_name.lower().replace("-", "_")
        in_lower = in_name.lower().replace("-", "_")

        compatible = _COMPATIBLE_TYPES.get(out_type, {out_type})
        if in_type not in compatible:
            return False

        if out_lower == in_lower:
            return True

        # Synthetic GET "id" only matches *_id params when the source tool suggests that entity
        if out_lower == "id" and in_lower.endswith("_id"):
            entity = in_lower[: -len("_id")]
            if not entity:
                return False
            source_lower = source_fqn.lower()
            return entity in source_lower

        if in_lower.endswith(f"_{out_lower}") or out_lower.endswith(f"_{in_lower}"):
            return True

        if _is_id_like(out_lower) and _is_id_like(in_lower):
            return True

        return False

    def neighbors(self, fqn: str, relation: str | None = None) -> list[str]:
        """Get neighbors of a node, optionally filtered by relation type."""
        result = []
        for _, dst, data in self.graph.out_edges(fqn, data=True):
            if relation is None or data.get("relation") == relation:
                result.append(dst)
        return result

    def get_node_data(self, fqn: str) -> dict[str, Any]:
        if fqn in self.graph:
            return dict(self.graph.nodes[fqn])
        return {}

    def get_endpoints_by_category(self, category: str) -> list[str]:
        return self._category_nodes.get(category, [])

    def get_endpoints_by_tool(self, tool_name: str) -> list[str]:
        return self._tool_nodes.get(tool_name, [])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved tool graph to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ToolGraph":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        g = cls()
        g.graph = nx.node_link_graph(data)
        for node, ndata in g.graph.nodes(data=True):
            tool_name = ndata.get("tool_name", node.split("/")[0])
            category = ndata.get("category", "Other")
            g._tool_nodes[tool_name].append(node)
            g._category_nodes[category].append(node)
        logger.info(f"Loaded tool graph: {g.num_nodes} nodes, {g.num_edges} edges")
        return g

    def summary(self) -> dict:
        relations = defaultdict(int)
        data_flow_edges = 0
        for _, _, data in self.graph.edges(data=True):
            relations[data.get("relation", "unknown")] += 1
            if data.get("data_flow"):
                data_flow_edges += 1
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_tools": len(self._tool_nodes),
            "num_categories": len(self._category_nodes),
            "edge_types": dict(relations),
            "data_flow_edges": data_flow_edges,
        }
