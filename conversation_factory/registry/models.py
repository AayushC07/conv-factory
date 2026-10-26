"""Canonical data models for the tool registry.

ToolBench raw JSON is inconsistent — missing fields, varying schemas, etc.
These Pydantic models normalize everything into a clean, uniform representation
that downstream components (graph, sampler, executor) can rely on.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ParamLocation(str, Enum):
    QUERY = "query"
    BODY = "body"
    PATH = "path"
    HEADER = "header"


class ParamType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class Parameter(BaseModel):
    """A single parameter for an API endpoint."""

    name: str
    description: str = ""
    param_type: ParamType = ParamType.STRING
    required: bool = False
    default: Any = None
    enum_values: list[str] = Field(default_factory=list)
    location: ParamLocation = ParamLocation.QUERY
    example: Any = None

    @field_validator("param_type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> ParamType:
        if isinstance(v, ParamType):
            return v
        mapping = {
            "str": ParamType.STRING,
            "int": ParamType.INTEGER,
            "float": ParamType.NUMBER,
            "num": ParamType.NUMBER,
            "bool": ParamType.BOOLEAN,
            "list": ParamType.ARRAY,
            "dict": ParamType.OBJECT,
        }
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in mapping:
                return mapping[v_lower]
            try:
                return ParamType(v_lower)
            except ValueError:
                return ParamType.STRING
        return ParamType.STRING


class ResponseField(BaseModel):
    """A field in an endpoint's response schema."""

    name: str
    field_type: ParamType = ParamType.STRING
    description: str = ""
    is_array: bool = False


class Endpoint(BaseModel):
    """A single API endpoint within a tool."""

    name: str
    description: str = ""
    method: str = "GET"
    path: str = ""
    parameters: list[Parameter] = Field(default_factory=list)
    response_schema: list[ResponseField] = Field(default_factory=list)
    response_description: str = ""

    @property
    def required_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters if p.required]

    @property
    def optional_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters if not p.required]

    @property
    def endpoint_id(self) -> str:
        """Stable identifier for this endpoint."""
        return f"{self.name}"


class ToolCategory(BaseModel):
    """A category grouping related tools."""

    name: str
    description: str = ""

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else str(v)


class Tool(BaseModel):
    """A complete tool with its endpoints, normalized from ToolBench data."""

    tool_name: str
    tool_id: str = ""
    description: str = ""
    category: ToolCategory = Field(default_factory=lambda: ToolCategory(name="Other"))
    endpoints: list[Endpoint] = Field(default_factory=list)
    home_url: str = ""
    api_host: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.tool_id:
            self.tool_id = hashlib.md5(
                self.tool_name.encode()
            ).hexdigest()[:12]

    @property
    def full_endpoint_ids(self) -> list[str]:
        """Return fully qualified endpoint identifiers: tool_name/endpoint_name."""
        return [f"{self.tool_name}/{ep.name}" for ep in self.endpoints]

    def get_endpoint(self, name: str) -> Optional[Endpoint]:
        for ep in self.endpoints:
            if ep.name == name:
                return ep
        return None
