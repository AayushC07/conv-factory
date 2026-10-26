"""Offline tool execution via LLM-generated or schema-derived mock responses.

Two strategies:
1. Schema-derived: generate responses from the endpoint's response_schema
   and parameter types — deterministic and fast.
2. LLM-generated: ask an LLM to produce a realistic response given the
   endpoint description, arguments, and conversation context — higher quality
   but requires API calls.

Session state ensures chain-consistent values: IDs returned by step N
are available to step N+1.
"""

from __future__ import annotations

import json
import logging
import random
import string
from typing import Any, Optional

from ..registry.models import Endpoint, ParamType, Tool
from ..registry.registry import ToolRegistry
from ..utils.llm import LLMClient
from .session import SessionState

logger = logging.getLogger(__name__)


class MockExecutor:
    """Produces mock tool responses consistent with endpoint schemas and session state."""

    def __init__(
        self,
        registry: ToolRegistry,
        llm_client: Optional[LLMClient] = None,
        use_llm: bool = True,
        seed: int | None = None,
    ) -> None:
        self.registry = registry
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None
        self.rng = random.Random(seed)

    async def execute(
        self,
        endpoint_fqn: str,
        arguments: dict[str, Any],
        session: SessionState,
    ) -> dict[str, Any]:
        """Execute a tool call and return a mock response."""
        result = self.registry.get_endpoint(endpoint_fqn)

        if result is None:
            # Unknown endpoint — generate a generic response
            response = self._generic_response(endpoint_fqn, arguments)
        else:
            tool, endpoint = result
            if self.use_llm:
                response = await self._llm_response(tool, endpoint, arguments, session)
            else:
                response = self._schema_response(endpoint, arguments, session)

        session.record_call(endpoint_fqn, arguments, response)
        return response

    def _schema_response(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        session: SessionState,
    ) -> dict[str, Any]:
        """Generate a response using keyword-based templates.

        The response_schema exists primarily for graph data-flow edges.
        The keyword templates in _infer_response produce richer, more
        realistic output (e.g. echoing input args, generating proper
        search result arrays) so we always use them for execution.
        """
        return self._infer_response(endpoint, arguments, session)

    async def _llm_response(
        self,
        tool: Tool,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        session: SessionState,
    ) -> dict[str, Any]:
        """Use LLM to generate a realistic response."""
        context = session.get_context_summary()

        prompt = (
            f"Generate a realistic JSON response for this API call.\n\n"
            f"Tool: {tool.tool_name} - {tool.description}\n"
            f"Endpoint: {endpoint.name} - {endpoint.description}\n"
            f"Method: {endpoint.method}\n"
            f"Arguments: {json.dumps(arguments)}\n"
            f"Previous calls in this session:\n{context}\n\n"
            f"Requirements:\n"
            f"- Return ONLY valid JSON (no markdown, no explanation)\n"
            f"- Use realistic but fictional data\n"
            f"- If this call depends on previous results, use values from previous responses\n"
            f"- Include appropriate status fields (success/error)\n"
            f"- Generate 2-5 results for search/list endpoints\n"
        )

        try:
            response = await self.llm_client.complete_structured(
                prompt, temperature=0.7
            )
            if isinstance(response, dict) and response:
                return response
            return self._schema_response(endpoint, arguments, session)
        except Exception as e:
            logger.warning(f"LLM mock response failed: {e}, falling back to schema")
            return self._schema_response(endpoint, arguments, session)

    def _infer_response(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        session: SessionState,
    ) -> dict[str, Any]:
        """Infer a plausible response when no schema is available."""
        method = endpoint.method.upper()
        name_lower = endpoint.name.lower()

        if any(kw in name_lower for kw in ("search", "list", "find", "get_all", "query")):
            return self._search_response(endpoint, arguments, session)
        elif any(kw in name_lower for kw in ("book", "create", "add", "register", "post")):
            return self._create_response(endpoint, arguments, session)
        elif any(kw in name_lower for kw in ("update", "edit", "modify", "put", "patch")):
            return self._update_response(endpoint, arguments, session)
        elif any(kw in name_lower for kw in ("delete", "remove", "cancel")):
            return self._delete_response(endpoint, arguments, session)
        elif any(kw in name_lower for kw in ("get", "fetch", "retrieve", "detail")):
            return self._detail_response(endpoint, arguments, session)
        elif method == "POST":
            return self._create_response(endpoint, arguments, session)
        elif method == "GET":
            return self._detail_response(endpoint, arguments, session)
        else:
            return {"status": "success", "message": f"{endpoint.name} completed"}

    def _search_response(
        self, endpoint: Endpoint, arguments: dict[str, Any], session: SessionState
    ) -> dict[str, Any]:
        n_results = self.rng.randint(2, 5)
        results = []
        for i in range(n_results):
            item = {"id": session.generate_id("item")}
            for param in endpoint.parameters:
                if param.name in arguments:
                    item[param.name] = arguments[param.name]
            item["name"] = f"Result {i + 1}"
            item["score"] = round(self.rng.uniform(0.5, 1.0), 2)
            results.append(item)
        return {"results": results, "total_count": n_results}

    def _create_response(
        self, endpoint: Endpoint, arguments: dict[str, Any], session: SessionState
    ) -> dict[str, Any]:
        prefix = endpoint.name.split("_")[0] if "_" in endpoint.name else "res"
        return {
            "id": session.generate_id(prefix),
            "status": "created",
            **{k: v for k, v in arguments.items() if not k.startswith("_")},
            "created_at": "2026-04-07T10:00:00Z",
        }

    def _update_response(
        self, endpoint: Endpoint, arguments: dict[str, Any], session: SessionState
    ) -> dict[str, Any]:
        return {
            "status": "updated",
            **{k: v for k, v in arguments.items()},
            "updated_at": "2026-04-07T10:30:00Z",
        }

    def _delete_response(
        self, endpoint: Endpoint, arguments: dict[str, Any], session: SessionState
    ) -> dict[str, Any]:
        return {"status": "deleted", "message": f"Successfully deleted"}

    def _detail_response(
        self, endpoint: Endpoint, arguments: dict[str, Any], session: SessionState
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"id": arguments.get("id", session.generate_id("item"))}
        for param in endpoint.parameters:
            if param.name in arguments:
                response[param.name] = arguments[param.name]
        response["name"] = f"Detail for {endpoint.name}"
        response["description"] = f"Detailed information retrieved via {endpoint.name}"
        return response

    def _generic_response(
        self, endpoint_fqn: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "endpoint": endpoint_fqn,
            "data": arguments,
        }

    def _generate_field_value(
        self,
        name: str,
        field_type: ParamType,
        is_array: bool,
        session: SessionState,
    ) -> Any:
        """Generate a plausible value for a response field."""
        # Check if session has a matching value
        existing = session.get_value(name)
        if existing is not None:
            return existing

        value = self._random_value(name, field_type)
        if is_array:
            return [self._random_value(name, field_type) for _ in range(self.rng.randint(2, 5))]
        return value

    def _random_value(self, name: str, field_type: ParamType) -> Any:
        name_lower = name.lower()
        if "id" in name_lower:
            prefix = name_lower.replace("_id", "").replace("id", "item")
            return f"{prefix}_{self.rng.randint(100, 9999)}"
        if field_type == ParamType.STRING:
            if "name" in name_lower:
                return self.rng.choice(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"])
            if "email" in name_lower:
                return "user@example.com"
            if "url" in name_lower or "link" in name_lower:
                return "https://example.com/resource"
            if "date" in name_lower or "time" in name_lower:
                return "2026-04-07T10:00:00Z"
            if "status" in name_lower:
                return "active"
            if "description" in name_lower:
                return "A detailed description."
            return "".join(self.rng.choices(string.ascii_lowercase, k=8))
        elif field_type == ParamType.INTEGER:
            return self.rng.randint(1, 1000)
        elif field_type == ParamType.NUMBER:
            return round(self.rng.uniform(0.0, 1000.0), 2)
        elif field_type == ParamType.BOOLEAN:
            return self.rng.choice([True, False])
        elif field_type == ParamType.ARRAY:
            return [self.rng.randint(1, 100) for _ in range(3)]
        elif field_type == ParamType.OBJECT:
            return {"key": "value"}
        return "unknown"
