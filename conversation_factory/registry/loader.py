"""ToolBench data loader.

Handles the inconsistencies in ToolBench raw JSON:
- Missing fields (description, parameters, etc.)
- Varying parameter formats (list vs dict, missing types)
- Inconsistent naming conventions
- Nested/flat tool definitions
- Uppercase type strings (NUMBER, STRING, BOOLEAN)
- template_response as an alternative response schema
- standardized_name as an alternative tool name
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .models import (
    Endpoint,
    Parameter,
    ParamLocation,
    ParamType,
    ResponseField,
    Tool,
    ToolCategory,
)

logger = logging.getLogger(__name__)


class ToolBenchLoader:
    """Loads and normalizes ToolBench tool definitions."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load_all(self, max_tools: int | None = None) -> list[Tool]:
        """Load all tools from the ToolBench data directory.

        Supports two layouts:
        1. Single file: data_dir contains a JSON file with all tools
        2. Directory of categories: data_dir/category/tool_name.json
        """
        tools: list[Tool] = []

        # Try single-file format first (common ToolBench export)
        json_files = list(self.data_dir.glob("*.json"))
        for jf in json_files:
            try:
                loaded = self._load_file(jf)
                tools.extend(loaded)
            except Exception as e:
                logger.warning(f"Failed to load {jf}: {e}")

        # Try category-directory format
        for category_dir in sorted(self.data_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category_name = category_dir.name
            for tool_file in sorted(category_dir.glob("*.json")):
                try:
                    loaded = self._load_tool_file(tool_file, category_name)
                    if loaded:
                        tools.append(loaded)
                except Exception as e:
                    logger.warning(f"Failed to load {tool_file}: {e}")

            if max_tools and len(tools) >= max_tools:
                break

        if max_tools:
            tools = tools[:max_tools]

        logger.info(f"Loaded {len(tools)} tools with {sum(len(t.endpoints) for t in tools)} endpoints")
        return tools

    def _load_file(self, path: Path) -> list[Tool]:
        """Load a JSON file that may contain one or many tools."""
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [self._parse_tool(item) for item in data if self._is_valid_tool(item)]
        elif isinstance(data, dict):
            # Check common wrapper keys
            for key in ("tools", "data", "results"):
                if key in data and isinstance(data[key], list):
                    tools = [self._parse_tool(item) for item in data[key] if self._is_valid_tool(item)]
                    if tools:
                        return tools
            if self._is_valid_tool(data):
                return [self._parse_tool(data)]
        return []

    def _load_tool_file(self, path: Path, category: str) -> Tool | None:
        """Load a single tool file within a category directory."""
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        if not self._is_valid_tool(data):
            return None

        tool = self._parse_tool(data)
        if not tool.category or tool.category.name == "Other":
            tool.category = ToolCategory(name=category)
        return tool

    def _is_valid_tool(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        has_name = any(k in data for k in (
            "tool_name", "name", "title", "standardized_name",
        ))
        has_apis = any(k in data for k in (
            "api_list", "apis", "endpoints", "functions",
        ))
        return has_name and has_apis

    def _parse_tool(self, data: dict[str, Any]) -> Tool:
        """Parse a raw tool dict into our canonical model."""
        name = (
            data.get("tool_name")
            or data.get("standardized_name")
            or data.get("name")
            or data.get("title", "unknown")
        )
        description = data.get("tool_description") or data.get("description") or ""
        category_name = (
            data.get("category_name")
            or data.get("category")
            or "Other"
        )
        home_url = data.get("home_url") or data.get("url") or ""
        api_host = data.get("host") or data.get("api_host") or ""

        raw_apis = (
            data.get("api_list")
            or data.get("apis")
            or data.get("endpoints")
            or data.get("functions")
            or []
        )

        endpoints = []
        for raw_ep in raw_apis:
            try:
                ep = self._parse_endpoint(raw_ep)
                if ep:
                    endpoints.append(ep)
            except Exception as e:
                logger.debug(f"Skipping endpoint in {name}: {e}")

        return Tool(
            tool_name=name,
            description=description,
            category=ToolCategory(name=category_name),
            endpoints=endpoints,
            home_url=home_url,
            api_host=api_host,
        )

    def _parse_endpoint(self, data: dict[str, Any]) -> Endpoint | None:
        if not isinstance(data, dict):
            return None

        name = data.get("name") or data.get("api_name") or data.get("function_name")
        if not name:
            return None

        description = (
            data.get("api_description")
            or data.get("description")
            or data.get("summary")
            or ""
        )
        method = (data.get("method") or data.get("http_method") or "GET").upper()
        path = data.get("api_url") or data.get("path") or data.get("url") or ""

        parameters = self._parse_parameters(data)
        response_schema = self._parse_response_schema(data)
        response_desc = data.get("response_description") or ""

        # ToolBench template_response: a sample response dict used as schema hint
        if not response_schema:
            template = data.get("template_response") or {}
            if isinstance(template, dict) and template:
                response_schema = self._parse_template_response(template)

        # Infer response fields when raw data has none — enriches data-flow
        # edges in the tool graph so the sampler can chain realistically.
        if not response_schema:
            response_schema = self._infer_response_schema(name, method, parameters)

        return Endpoint(
            name=name,
            description=description,
            method=method,
            path=path,
            parameters=parameters,
            response_schema=response_schema,
            response_description=response_desc,
        )

    def _parse_parameters(self, data: dict[str, Any]) -> list[Parameter]:
        """Extract parameters from various ToolBench formats."""
        params: list[Parameter] = []

        raw_required = data.get("required_parameters") or []
        raw_optional = data.get("optional_parameters") or []

        for p in raw_required:
            parsed = self._parse_single_param(p, required=True)
            if parsed:
                params.append(parsed)

        for p in raw_optional:
            parsed = self._parse_single_param(p, required=False)
            if parsed:
                params.append(parsed)

        # Fallback: check for "parameters" field
        if not params:
            raw_params = data.get("parameters") or []
            if isinstance(raw_params, dict):
                for k, v in raw_params.items():
                    params.append(Parameter(
                        name=k,
                        description=str(v.get("description", "")) if isinstance(v, dict) else "",
                        param_type=v.get("type", "string") if isinstance(v, dict) else "string",
                        required=v.get("required", False) if isinstance(v, dict) else False,
                    ))
            elif isinstance(raw_params, list):
                for p in raw_params:
                    parsed = self._parse_single_param(p, required=False)
                    if parsed:
                        params.append(parsed)

        return params

    def _parse_single_param(self, data: Any, required: bool) -> Parameter | None:
        if isinstance(data, dict):
            name = data.get("name") or data.get("param_name")
            if not name:
                return None
            # ToolBench uses "default" for example values; skip empty-string defaults
            default = data.get("default")
            if isinstance(default, str) and default.strip() == "":
                default = None
            # "required" inside the param dict overrides the list-level flag
            if "required" in data:
                required = bool(data["required"])
            return Parameter(
                name=name,
                description=data.get("description") or "",
                param_type=data.get("type") or data.get("param_type") or "string",
                required=required,
                default=default,
                enum_values=data.get("enum") or data.get("enum_values") or [],
                example=data.get("example") or data.get("example_value"),
            )
        elif isinstance(data, str):
            return Parameter(name=data, required=required)
        return None

    def _parse_response_schema(self, data: dict[str, Any]) -> list[ResponseField]:
        """Best-effort extraction of response fields."""
        fields: list[ResponseField] = []

        schema = data.get("response_schema") or data.get("output_schema") or {}
        if isinstance(schema, dict):
            properties = schema.get("properties") or schema
            if not isinstance(properties, dict):
                return fields
            for fname, info in properties.items():
                if not isinstance(info, dict):
                    continue
                fields.append(ResponseField(
                    name=fname,
                    field_type=info.get("type", "string"),
                    description=info.get("description", ""),
                    is_array=info.get("type") == "array",
                ))
        elif isinstance(schema, list):
            # Some exports use a list of {name, type, description} objects
            for item in schema:
                if isinstance(item, dict) and "name" in item:
                    fields.append(ResponseField(
                        name=item["name"],
                        field_type=item.get("type", "string"),
                        description=item.get("description", ""),
                        is_array=item.get("type") == "array",
                    ))

        return fields

    @staticmethod
    def _parse_template_response(template: dict[str, Any]) -> list[ResponseField]:
        """Extract response fields from ToolBench's template_response.

        template_response is an example response dict like:
        {"results": [{"id": 1, "name": "..."}], "status": "ok"}
        We infer field types from the Python types of the values.
        """
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
        fields: list[ResponseField] = []
        for key, value in template.items():
            if not isinstance(key, str):
                continue
            py_type = type(value)
            field_type = type_map.get(py_type, "string")
            fields.append(ResponseField(
                name=key,
                field_type=field_type,
                is_array=isinstance(value, list),
            ))
        return fields

    @staticmethod
    def _infer_response_schema(
        name: str, method: str, parameters: list[Parameter]
    ) -> list[ResponseField]:
        """Heuristic response fields when ToolBench data has no response_schema.

        Enriches data-flow edges in the tool graph so the sampler can chain
        endpoints realistically (e.g. search → get_details → book).
        """
        lower = name.lower()
        fields: list[ResponseField] = []

        # Collect parameter names that hint at the domain entity
        param_names = {p.name.lower() for p in parameters}

        # --- search / list endpoints ---
        if any(tok in lower for tok in ("search", "list", "find", "browse", "explore", "query")):
            fields.append(ResponseField(name="results", field_type="array", is_array=True))
            fields.append(ResponseField(name="id", field_type="string"))
            fields.append(ResponseField(name="name", field_type="string"))
            fields.append(ResponseField(name="total_count", field_type="integer"))
            return fields

        # --- get / detail / info endpoints ---
        if any(tok in lower for tok in ("get", "detail", "info", "fetch", "show", "view", "retrieve")):
            fields.append(ResponseField(name="id", field_type="string"))
            fields.append(ResponseField(name="name", field_type="string"))
            fields.append(ResponseField(name="description", field_type="string"))
            # Mirror ID params back (e.g. hotel_id input → hotel_id in response)
            for p in parameters:
                if p.name.lower().endswith("_id") and p.name.lower() != "id":
                    fields.append(ResponseField(name=p.name, field_type="string"))
            return fields

        # --- book / create / reserve / submit / order endpoints ---
        if any(tok in lower for tok in ("book", "create", "reserve", "submit", "order", "purchase", "register")):
            fields.append(ResponseField(name="id", field_type="string"))
            fields.append(ResponseField(name="status", field_type="string"))
            fields.append(ResponseField(name="confirmation_code", field_type="string"))
            return fields

        # --- update / modify / edit endpoints ---
        if any(tok in lower for tok in ("update", "modify", "edit", "patch")):
            fields.append(ResponseField(name="id", field_type="string"))
            fields.append(ResponseField(name="status", field_type="string"))
            fields.append(ResponseField(name="updated", field_type="boolean"))
            return fields

        # --- delete / cancel / remove endpoints ---
        if any(tok in lower for tok in ("delete", "cancel", "remove")):
            fields.append(ResponseField(name="status", field_type="string"))
            fields.append(ResponseField(name="success", field_type="boolean"))
            return fields

        # --- fallback: POST implies creation-like, GET implies read-like ---
        if method == "POST":
            fields.append(ResponseField(name="id", field_type="string"))
            fields.append(ResponseField(name="status", field_type="string"))
        elif method in ("GET", ""):
            fields.append(ResponseField(name="id", field_type="string"))
            fields.append(ResponseField(name="name", field_type="string"))

        return fields
