"""Tool Registry — the central lookup for all normalized tools and endpoints."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .models import Endpoint, Tool, ToolCategory

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Indexed, queryable registry of all loaded tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._by_category: dict[str, list[str]] = defaultdict(list)
        # endpoint_fqn -> (tool_name, endpoint_name)
        self._endpoint_index: dict[str, tuple[str, str]] = {}

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def categories(self) -> list[str]:
        return sorted(self._by_category.keys())

    @property
    def num_tools(self) -> int:
        return len(self._tools)

    @property
    def num_endpoints(self) -> int:
        return len(self._endpoint_index)

    def add_tool(self, tool: Tool) -> None:
        self._tools[tool.tool_name] = tool
        self._by_category[tool.category.name].append(tool.tool_name)
        for ep in tool.endpoints:
            fqn = f"{tool.tool_name}/{ep.name}"
            self._endpoint_index[fqn] = (tool.tool_name, ep.name)

    def add_tools(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.add_tool(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_endpoint(self, fqn: str) -> Optional[tuple[Tool, Endpoint]]:
        """Look up a fully-qualified endpoint name like 'tool_name/endpoint_name'."""
        if fqn not in self._endpoint_index:
            return None
        tool_name, ep_name = self._endpoint_index[fqn]
        tool = self._tools[tool_name]
        ep = tool.get_endpoint(ep_name)
        if ep is None:
            return None
        return (tool, ep)

    def get_tools_by_category(self, category: str) -> list[Tool]:
        tool_names = self._by_category.get(category, [])
        return [self._tools[n] for n in tool_names if n in self._tools]

    def get_all_endpoint_fqns(self) -> list[str]:
        return list(self._endpoint_index.keys())

    def search_tools(self, query: str) -> list[Tool]:
        """Simple keyword search across tool names and descriptions."""
        query_lower = query.lower()
        results = []
        for tool in self._tools.values():
            if (query_lower in tool.tool_name.lower()
                    or query_lower in tool.description.lower()):
                results.append(tool)
        return results

    def save(self, path: str | Path) -> None:
        """Serialize registry to JSON."""
        path = Path(path)
        data = {
            "tools": [t.model_dump(mode="json") for t in self._tools.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved registry with {self.num_tools} tools to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ToolRegistry":
        """Deserialize registry from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        registry = cls()
        for tool_data in data["tools"]:
            tool = Tool.model_validate(tool_data)
            registry.add_tool(tool)
        logger.info(f"Loaded registry with {registry.num_tools} tools from {path}")
        return registry

    def summary(self) -> dict:
        return {
            "num_tools": self.num_tools,
            "num_endpoints": self.num_endpoints,
            "num_categories": len(self._by_category),
            "categories": {cat: len(tools) for cat, tools in self._by_category.items()},
        }
