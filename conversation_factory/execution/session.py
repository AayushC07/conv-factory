"""Session state management for multi-step tool execution.

Maintains a running context of all tool calls and their outputs within
a conversation, so that later calls can reference values returned by
earlier calls (e.g., IDs, tokens, lists of results).
"""

from __future__ import annotations

import copy
import re
from typing import Any


class SessionState:
    """Tracks tool execution history and extractable values within a conversation."""

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []
        self._value_store: dict[str, Any] = {}
        self._id_counter: int = 0

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def values(self) -> dict[str, Any]:
        return dict(self._value_store)

    def record_call(
        self, endpoint: str, arguments: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Record a tool call and index its output values for later reference."""
        entry = {
            "step": len(self._history),
            "endpoint": endpoint,
            "arguments": copy.deepcopy(arguments),
            "response": copy.deepcopy(response),
        }
        self._history.append(entry)
        self._index_response(endpoint, response)

    def _index_response(self, endpoint: str, response: Any, prefix: str = "") -> None:
        """Recursively index response values for later retrieval."""
        if isinstance(response, dict):
            for key, value in response.items():
                full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
                self._value_store[full_key] = value
                if prefix == "":
                    ep_key = f"{endpoint}/{key}"
                    self._value_store[ep_key] = value
                self._index_response(endpoint, value, full_key)
        elif isinstance(response, list) and response:
            self._value_store[prefix] = response
            if isinstance(response[0], dict):
                for key in response[0]:
                    values = [item.get(key) for item in response if isinstance(item, dict) and key in item]
                    list_key = f"{prefix}.*.{key}" if prefix else f"*.{key}"
                    self._value_store[list_key] = values

    def get_value(self, key: str) -> Any:
        return self._value_store.get(key)

    def get_available_ids(self) -> dict[str, list[Any]]:
        """Return all ID-like values from previous responses, grouped by key."""
        ids: dict[str, list[Any]] = {}
        for key, value in self._value_store.items():
            if self._is_id_key(key):
                if isinstance(value, list):
                    ids[key] = value
                else:
                    ids[key] = [value]
        return ids

    def get_last_response(self) -> dict[str, Any] | None:
        if self._history:
            return self._history[-1].get("response")
        return None

    def get_response_for_endpoint(self, endpoint: str) -> dict[str, Any] | None:
        """Get the most recent response for a given endpoint."""
        for entry in reversed(self._history):
            if entry["endpoint"] == endpoint:
                return entry.get("response")
        return None

    def generate_id(self, prefix: str = "id") -> str:
        """Generate a unique mock ID."""
        self._id_counter += 1
        return f"{prefix}_{self._id_counter:04d}"

    def get_context_summary(self) -> str:
        """Human-readable summary of session state for prompt injection."""
        if not self._history:
            return "No previous tool calls."

        lines = []
        for entry in self._history:
            ep = entry["endpoint"]
            args = entry["arguments"]
            resp = entry["response"]
            resp_summary = self._summarize_value(resp)
            lines.append(f"Step {entry['step']}: Called {ep}({args}) -> {resp_summary}")
        return "\n".join(lines)

    @staticmethod
    def _is_id_key(key: str) -> bool:
        lower = key.lower()
        # Match ID-like suffixes/words but avoid false positives like
        # "phone_number", "street_number", "total_count".
        parts = lower.replace("-", "_").replace(".", "_").split("_")
        id_tokens = {"id", "ids", "identifier", "key", "code", "ref", "token"}
        if any(p in id_tokens for p in parts):
            return True
        # "booking_number" is an ID; "phone_number" is not — only match
        # *_number when the prefix suggests an identifier entity.
        if "number" in parts:
            id_prefixes = {"booking", "confirmation", "order", "reservation", "tracking", "reference"}
            return bool(set(parts) & id_prefixes)
        return False

    @staticmethod
    def _summarize_value(value: Any, max_len: int = 200) -> str:
        import json
        try:
            s = json.dumps(value, default=str)
        except (TypeError, ValueError):
            s = str(value)
        if len(s) > max_len:
            return s[:max_len] + "..."
        return s

    def reset(self) -> None:
        self._history.clear()
        self._value_store.clear()
        self._id_counter = 0
