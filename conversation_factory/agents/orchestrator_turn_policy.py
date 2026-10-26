"""Turn-level policies for the conversation orchestrator (e.g. booking gates)."""

from __future__ import annotations

from typing import Any

from ..registry.endpoint_semantics import (
    is_likely_commit_operation,
    is_likely_listing_search_operation,
)

from .assistant_agent import AssistantAction


def action_includes_booking(action: AssistantAction) -> bool:
    """True if any tool call looks like a commit mutation (book, pay, create, etc.)."""
    for tc in action.tool_calls:
        ep = tc.get("endpoint", "")
        if is_likely_commit_operation(ep):
            return True
    return False


def user_message_after_last_listing_search(messages: list[dict[str, Any]]) -> bool:
    """True if the user spoke at least once after the latest listing-search tool batch.

    If there has been no listing search yet, returns True (no gating).
    """
    n = len(messages)
    i = n - 1
    while i >= 0:
        if messages[i].get("role") != "tool":
            i -= 1
            continue
        batch_end = i
        batch_start = i
        while batch_start > 0 and messages[batch_start - 1].get("role") == "tool":
            batch_start -= 1
        asst_idx = batch_start - 1
        if asst_idx < 0:
            i = batch_start - 1
            continue
        asst = messages[asst_idx]
        if asst.get("role") != "assistant":
            i = batch_start - 1
            continue
        had_listing_search = False
        for tc in asst.get("tool_calls") or []:
            ep = tc.get("endpoint", "")
            if is_likely_listing_search_operation(ep):
                had_listing_search = True
                break
        if had_listing_search:
            for j in range(batch_end + 1, n):
                if messages[j].get("role") == "user":
                    return True
            return False
        i = batch_start - 1
    return True


def clarify_before_booking_action() -> AssistantAction:
    """Replace a premature commit attempt with a clarification turn."""
    return AssistantAction(
        action_type="ask_clarification",
        message=(
            "Which option from the results do you want? Name the listing or tell me the "
            "number from the list, and I can proceed."
        ),
        tool_calls=[],
        reasoning="policy: need explicit user choice after listing search results",
    )
