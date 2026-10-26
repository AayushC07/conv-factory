"""Validate full message transcripts (e.g. after LLM repair) by replaying tool-call rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..execution.session import SessionState
from ..execution.tool_call_validation import validate_before_execute

if TYPE_CHECKING:
    from ..registry.registry import ToolRegistry


def replay_validate_conversation_messages(
    messages: list[dict[str, Any]],
    *,
    registry: Optional["ToolRegistry"] = None,
) -> list[str]:
    """Replay tool calls in order and collect the same validation errors as live execution.

    Use after repair so invalid tool arguments cannot appear with fake success payloads.
    """
    session = SessionState()
    errors: list[str] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tcs = msg["tool_calls"]
            if not isinstance(tcs, list):
                i += 1
                continue
            i += 1
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                if i >= len(messages) or messages[i].get("role") != "tool":
                    errors.append(
                        "Malformed transcript: expected one tool message per tool call after assistant tool_calls."
                    )
                    return errors
                ep = (tc.get("endpoint") or "").strip()
                args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                err = validate_before_execute(ep, args, session, registry=registry)
                tool_content = messages[i].get("content")
                if not isinstance(tool_content, dict):
                    tool_content = {"_non_dict": tool_content}
                if err is not None:
                    detail = err.get("error", err) if isinstance(err, dict) else err
                    errors.append(f"{ep}: {detail}")
                    session.record_call(ep, args, err)
                else:
                    session.record_call(ep, args, tool_content)
                i += 1
            continue
        i += 1
    return errors


def _extract_direction_instructions(payload: dict[str, Any]) -> list[str]:
    """Collect turn instruction strings from common Maps get_directions response shapes."""
    if not isinstance(payload, dict):
        return []

    candidates: list[Any] = []
    for path in (
        ("steps",),
        ("data", "steps"),
        ("directions",),
        ("data", "directions"),
    ):
        cur: Any = payload
        for p in path:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                cur = None
                break
        if isinstance(cur, list):
            candidates.append(cur)

    dirs = payload.get("directions")
    if isinstance(dirs, dict) and isinstance(dirs.get("steps"), list):
        candidates.append(dirs["steps"])

    out: list[str] = []
    for cur in candidates:
        for item in cur:
            if isinstance(item, dict) and isinstance(item.get("instruction"), str):
                t = item["instruction"].strip()
                if t:
                    out.append(t)
        if out:
            return out
    return out


def validate_directions_grounding(messages: list[dict[str, Any]]) -> list[str]:
    """Ensure the first user-facing assistant reply after get_directions includes tool step text."""
    errors: list[str] = []
    pending: list[str] | None = None

    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tcs = msg["tool_calls"]
            i += 1
            if not isinstance(tcs, list):
                continue
            for tc in tcs:
                if i >= len(messages):
                    break
                if messages[i].get("role") != "tool":
                    break
                if not isinstance(tc, dict):
                    i += 1
                    continue
                ep = (tc.get("endpoint") or "").lower()
                content = messages[i].get("content")
                if isinstance(content, dict) and "get_directions" in ep:
                    steps = _extract_direction_instructions(content)
                    if steps:
                        pending = steps
                i += 1
            continue

        if msg.get("role") == "assistant" and msg.get("content") and not msg.get("tool_calls"):
            text = str(msg.get("content") or "")
            if pending:
                lower = text.lower()
                missing: list[str] = []
                for instr in pending:
                    instr = instr.strip()
                    if not instr:
                        continue
                    prefix = instr[: min(36, len(instr))].lower()
                    if prefix not in lower:
                        missing.append(instr[:72])
                if missing:
                    errors.append(
                        "Assistant must copy turn-by-turn instructions from Maps_API/get_directions "
                        f"(tool JSON steps), not invent different roads or exits. Missing expected step text "
                        f"(prefixes): {missing[:3]!r}"
                    )
                pending = None
            i += 1
            continue
        i += 1
    return errors


def validate_repaired_transcript(
    messages: list[dict[str, Any]],
    *,
    registry: Optional["ToolRegistry"] = None,
) -> list[str]:
    """Run replay validation and directions checks; return all error strings."""
    out = replay_validate_conversation_messages(messages, registry=registry)
    out.extend(validate_directions_grounding(messages))
    return out
