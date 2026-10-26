"""Structural validation of generated conversations (runaway tools, etc.)."""

from __future__ import annotations

import json
from typing import Any


def _tool_call_signature(endpoint: str, arguments: dict[str, Any]) -> str:
    return f"{endpoint}|{json.dumps(arguments, sort_keys=True, default=str)}"


def max_consecutive_identical_tool_calls(messages: list[dict[str, Any]]) -> int:
    """Longest run of identical (endpoint, arguments) tool calls in message order."""
    best = 0
    cur = 0
    last_sig: str | None = None
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            ep = tc.get("endpoint", "")
            args = tc.get("arguments")
            if not isinstance(args, dict):
                args = {}
            sig = _tool_call_signature(ep, args)
            if sig == last_sig:
                cur += 1
            else:
                cur = 1
                last_sig = sig
            best = max(best, cur)
    return best


def has_redundant_tool_runaway(
    messages: list[dict[str, Any]], *, threshold: int = 3
) -> bool:
    """True if the same tool+args appears at least `threshold` times in a row."""
    return max_consecutive_identical_tool_calls(messages) >= threshold
