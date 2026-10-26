"""Normalize repaired/exported conversations to the orchestrator message schema."""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from ..registry.endpoint_semantics import resolve_endpoint_fqn
from ..registry.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Re-export for callers that normalize messages alongside registry-aware resolution.
__all__ = [
    "normalize_conversation",
    "normalize_messages",
    "resolve_endpoint_fqn",
    "sync_metadata_from_messages",
    "sanitize_messages_for_export",
    "validate_messages_schema",
    "ensure_closing_assistant_after_tool_tail",
]

_ALLOWED_ROLES = frozenset({"user", "assistant", "tool", "system"})

# Repair LLM output that used a role name as the JSON key instead of role/content.
_ROLE_KEY_TYPOS = ("ASSISTANT", "USER", "TOOL", "SYSTEM")

# LLM-hallucinated role values → correct role.
# The LLM sometimes invents roles like "tool_call", "tool_calls", "function", etc.
_ROLE_CORRECTIONS: dict[str, str] = {
    "tool_call": "assistant",
    "tool_calls": "assistant",
    "function": "assistant",
    "function_call": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "agent": "assistant",
    "human": "user",
    "customer": "user",
    "tool_result": "tool",
    "tool_response": "tool",
    "function_response": "tool",
    "observation": "tool",
}


def sanitize_messages_for_export(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fix common malformed message dicts before normalization.

    Handles e.g. ``{\"ASSISTANT\": \"...\"}`` → ``{\"role\": \"assistant\", \"content\": \"...\"}``.
    Returns (fixed_messages, warning_strings).
    """
    fixed: list[dict[str, Any]] = []
    warnings: list[str] = []
    for i, raw in enumerate(messages):
        if not isinstance(raw, dict):
            warnings.append(
                f"messages[{i}]: expected dict, got {type(raw).__name__!r}, skipped"
            )
            continue
        m = dict(raw)
        role_val = m.get("role")
        if role_val is None or (isinstance(role_val, str) and not role_val.strip()):
            for typo in _ROLE_KEY_TYPOS:
                if typo in m:
                    val = m.pop(typo)
                    m["role"] = typo.lower()
                    m["content"] = val
                    warnings.append(
                        f"messages[{i}]: fixed key {typo!r} → role={m['role']!r} and content"
                    )
                    break
        role = m.get("role")
        if isinstance(role, str):
            low = role.lower().strip()
            # Fix casing (e.g. "Assistant" → "assistant")
            if low in _ALLOWED_ROLES and role != low:
                m["role"] = low
                warnings.append(f"messages[{i}]: normalized role casing {role!r} → {low!r}")
            # Fix hallucinated roles (e.g. "tool_call" → "assistant")
            elif low in _ROLE_CORRECTIONS:
                corrected = _ROLE_CORRECTIONS[low]
                m["role"] = corrected
                warnings.append(
                    f"messages[{i}]: corrected hallucinated role {role!r} → {corrected!r}"
                )
            elif low not in _ALLOWED_ROLES:
                # Unknown role — drop the message entirely to avoid corrupt training data
                warnings.append(
                    f"messages[{i}]: dropped message with unrecognized role {role!r}"
                )
                continue
        fixed.append(m)
    return fixed, warnings


def validate_messages_schema(messages: list[dict[str, Any]]) -> list[str]:
    """Return human-readable errors if ``messages`` violate the export schema."""
    errors: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if not isinstance(role, str) or not role:
            errors.append(f"messages[{i}]: missing or invalid role")
            continue
        if role not in _ALLOWED_ROLES:
            errors.append(
                f"messages[{i}]: role {role!r} not in {sorted(_ALLOWED_ROLES)}"
            )
            continue
        if role == "assistant":
            has_tools = bool(msg.get("tool_calls"))
            content = msg.get("content")
            if not has_tools and content is None:
                errors.append(
                    f"messages[{i}]: assistant must have content or non-empty tool_calls"
                )
        elif role == "tool":
            if "content" not in msg:
                errors.append(f"messages[{i}]: tool message missing content key")
        elif role == "user":
            if "content" not in msg:
                errors.append(f"messages[{i}]: user message missing content key")
    return errors


_CLOSING_ASSISTANT_TEXT = (
    "I've incorporated the latest tool results. Let me know if you need anything else."
)


def ensure_closing_assistant_after_tool_tail(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Append a minimal assistant message if the transcript ends on ``tool`` role(s).

    Avoids orphan tool tails in JSONL exports (last message must not be only tool output).
    """
    if not messages:
        return messages, []
    if messages[-1].get("role") != "tool":
        return messages, []
    out = list(messages)
    out.append(
        {
            "role": "assistant",
            "content": _CLOSING_ASSISTANT_TEXT,
        }
    )
    return out, [
        "appended closing assistant message after terminal tool message(s)"
    ]


def _normalize_tool_call(
    tc: dict[str, Any], registry: Optional[ToolRegistry] = None
) -> dict[str, Any]:
    out = dict(tc)
    ep = out.get("endpoint") or out.get("tool") or ""
    ep = resolve_endpoint_fqn(str(ep).strip(), registry)
    out["endpoint"] = ep
    out.pop("tool", None)
    args = out.get("arguments")
    if args is None:
        args = out.get("args", {})
    if not isinstance(args, dict):
        args = {}
    out["arguments"] = args
    out.pop("args", None)
    return out


def normalize_messages(
    messages: list[dict[str, Any]], registry: Optional[ToolRegistry] = None
) -> list[dict[str, Any]]:
    """Ensure assistant ``tool_calls`` use ``endpoint`` / ``arguments`` and registered FQNs."""
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        m = copy.deepcopy(msg)
        tcs = m.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            m["tool_calls"] = [
                _normalize_tool_call(tc, registry) if isinstance(tc, dict) else tc
                for tc in tcs
            ]
            if m.get("role") == "assistant":
                m["content"] = None
        elif m.get("role") == "assistant" and isinstance(tcs, list) and not tcs:
            m.pop("tool_calls", None)
        normalized.append(m)
    return normalized


def sync_metadata_from_messages(
    conversation: dict[str, Any], registry: Optional[ToolRegistry] = None
) -> None:
    """Recompute tools_used, call counts, and num_turns from ``messages`` (post-repair)."""
    messages = conversation.get("messages") or []
    meta = conversation.setdefault("metadata", {})
    seen: set[str] = set()
    tools_ordered: list[str] = []
    num_tool_calls = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            ep = tc.get("endpoint") or tc.get("tool") or ""
            ep = resolve_endpoint_fqn(str(ep).strip(), registry) if ep else ""
            if not ep:
                continue
            num_tool_calls += 1
            if ep not in seen:
                seen.add(ep)
                tools_ordered.append(ep)
    meta["tools_used"] = tools_ordered
    meta["num_tool_calls"] = num_tool_calls
    meta["num_distinct_tools"] = len(tools_ordered)
    meta["num_turns"] = len(messages)


def normalize_conversation(
    conversation: dict[str, Any], registry: Optional[ToolRegistry] = None
) -> dict[str, Any]:
    """Sanitize export schema, normalize tool_calls, validate, and sync metadata."""
    conv = conversation
    cid = conv.get("conversation_id", "?")
    raw = conv.get("messages") or []
    sanitized, san_warnings = sanitize_messages_for_export(raw)
    for w in san_warnings:
        logger.warning("conversation %s: %s", cid, w)
    normalized = normalize_messages(sanitized, registry)
    normalized, tail_warnings = ensure_closing_assistant_after_tool_tail(normalized)
    for w in tail_warnings:
        logger.warning("conversation %s: %s", cid, w)
    all_fixes = [*san_warnings, *tail_warnings]
    schema_errors = validate_messages_schema(normalized)
    for e in schema_errors:
        logger.warning("conversation %s: schema %s", cid, e)
    conv["messages"] = normalized
    meta = conv.setdefault("metadata", {})
    meta.pop("schema_fixes", None)
    meta.pop("schema_validation_errors", None)
    meta.pop("orphan_tool_tail_fixed", None)
    if all_fixes:
        meta["schema_fixes"] = all_fixes
    if tail_warnings:
        meta["orphan_tool_tail_fixed"] = True
    if schema_errors:
        meta["schema_validation_errors"] = schema_errors
    sync_metadata_from_messages(conv, registry)
    return conv
