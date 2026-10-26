"""Within-conversation grounding.

Ensures that tool call arguments in step N reference real values from
step N-1 outputs rather than hallucinated placeholders. Works by:

1. Maintaining a SessionState of all prior outputs
2. Before each tool call, scanning arguments for ID-like fields and
   replacing placeholders with actual values from the session
3. Injecting session context into the assistant agent's prompt so the
   LLM is aware of available values
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..execution.session import SessionState
from ..graph.sampler import ToolChain

logger = logging.getLogger(__name__)


class ConversationGrounding:
    """Grounds tool call arguments in prior conversation outputs."""

    # Patterns that suggest a value is a placeholder rather than real data
    _PLACEHOLDER_PATTERNS = [
        re.compile(r"^\{.*\}$"),          # {some_id}
        re.compile(r"^<.*>$"),            # <hotel_id>
        re.compile(r"^placeholder", re.I),
        re.compile(r"^example", re.I),
        re.compile(r"^your_", re.I),
        re.compile(r"^TODO", re.I),
        re.compile(r"^xxx", re.I),
    ]

    def ground_arguments(
        self,
        endpoint: str,
        arguments: dict[str, Any],
        session: SessionState,
    ) -> dict[str, Any]:
        """Replace placeholder values with real values from session state."""
        grounded = dict(arguments)
        available_ids = session.get_available_ids()

        for key, value in grounded.items():
            if not isinstance(value, str):
                continue

            # Check if this looks like a placeholder
            if self._is_placeholder(value):
                replacement = self._find_replacement(key, available_ids, session)
                if replacement is not None:
                    grounded[key] = replacement
                    logger.debug(f"Grounded {key}: '{value}' -> '{replacement}'")

            # Check if this references a value from a known response
            elif self._is_id_key(key):
                existing = session.get_value(key)
                if existing is not None and existing != value:
                    # The session has a real value — prefer it if the arg looks fabricated
                    if not self._looks_real(value):
                        grounded[key] = existing

        return grounded

    def get_assistant_instruction(
        self,
        session: SessionState,
        tool_chain: ToolChain,
        current_turn: int,
        num_tool_calls: int = 0,
    ) -> str:
        """Generate grounding instructions for the assistant agent."""
        parts = []

        if session.history:
            ids = session.get_available_ids()
            if ids:
                id_summary = []
                for key, values in ids.items():
                    display_vals = values[:5]
                    id_summary.append(f"  {key}: {display_vals}")
                parts.append(
                    "IMPORTANT: Use these real values from previous tool responses "
                    "(do NOT make up IDs or use placeholders):\n"
                    + "\n".join(id_summary)
                )
                parts.append(
                    "When summarizing search or list results for the user, use ONLY names, prices, and fields "
                    "that appear in those tool responses — do not invent alternate hotels, flights, or numbers."
                )
                parts.append(
                    "Weather: cite only dates and temperatures present in the forecast objects. "
                    "Booking: do not call book_* or make_reservation until the user has picked a specific option "
                    "from the latest search results."
                )

        # Use actual tool call count (not turn index) to determine remaining tools.
        # This prevents the remaining list from shrinking due to clarification turns.
        called_count = min(num_tool_calls, len(tool_chain.endpoints))
        remaining = tool_chain.endpoints[called_count:]
        if remaining:
            urgency = ""
            if num_tool_calls == 0 and current_turn >= 2:
                urgency = (
                    " You MUST call the next tool now — do not respond with text only. "
                    "If you need a parameter value, ask the user in a clarification turn, "
                    "then call the tool on the very next turn."
                )
            parts.append(
                f"Remaining tools you MUST call in this conversation: {', '.join(remaining[:5])}.{urgency}"
            )

        domain = self._tool_chain_domain_hints(tool_chain)
        if domain:
            parts.append(domain)

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _tool_chain_domain_hints(tool_chain: ToolChain) -> str:
        """Extra rules when the sampled chain mixes domains that are often confused."""
        eps = tool_chain.endpoints
        lines: list[str] = []
        has_flights = any("Flights_API" in e for e in eps)
        has_events = any("Events_API" in e for e in eps)
        if has_flights and has_events:
            lines.append(
                "Flights vs events: use **flight_id** values only from **Flights_API/search_flights** "
                "responses for **Flights_API/book_flight**. Never pass **event** or **ticket** ids from "
                "**Events_API/search_events** into **book_flight** — use event-appropriate tools if the API provides them."
            )
        if any("Restaurant_API" in e for e in eps):
            lines.append(
                "Restaurants: when you list or describe venues, every **name**, **address**, and **price** "
                "must match the **Restaurant_API** JSON character-for-character (no invented alternates)."
            )
        if any("Hotels_API" in e for e in eps):
            lines.append(
                "Hotels: repeat **hotel names and prices** exactly as in **Hotels_API** tool JSON."
            )
        return "\n\n".join(lines) if lines else ""

    def _find_replacement(
        self,
        key: str,
        available_ids: dict[str, list[Any]],
        session: SessionState,
    ) -> Any:
        """Find a suitable replacement value from the session."""
        key_lower = key.lower()
        # Extract the core concept: "hotel_id" -> "id", "booking_ref" -> "ref"
        key_parts = key_lower.replace("-", "_").split("_")

        # Direct match
        if key in available_ids and available_ids[key]:
            return available_ids[key][0]

        # Fuzzy match on ID-like keys
        for id_key, values in available_ids.items():
            id_key_lower = id_key.lower()
            if key_lower in id_key_lower or id_key_lower in key_lower:
                if values:
                    return values[0]
            # Match if key's suffix (e.g. "id") appears as a suffix in the stored key
            id_key_parts = id_key_lower.replace(".", "_").replace("*", "").split("_")
            common = set(key_parts) & set(id_key_parts)
            if common and any(p in common for p in ("id", "ref", "key", "token", "code")):
                if values:
                    return values[0]

        # Check value store directly
        val = session.get_value(key)
        if val is not None:
            if isinstance(val, list) and val:
                return val[0]
            return val

        # Search all values for anything matching the key pattern
        for store_key, store_val in session.values.items():
            if key_lower in store_key.lower():
                if isinstance(store_val, list) and store_val:
                    return store_val[0]
                return store_val

        # Last resort: if the key looks like it wants an ID, find an ID from the
        # same domain (entity prefix).  Do NOT return IDs from unrelated domains
        # (e.g. flight_id for hotel_id) as that breaks coherent chaining.
        if self._is_id_key(key):
            entity = key_lower.replace("_id", "").replace("id", "").strip("_")
            if entity:
                for id_key, values in available_ids.items():
                    if entity in id_key.lower() and values:
                        return values[0]

        return None

    def _is_placeholder(self, value: str) -> bool:
        return any(p.match(str(value)) for p in self._PLACEHOLDER_PATTERNS)

    @staticmethod
    def _is_id_key(key: str) -> bool:
        lower = key.lower()
        return any(p in lower for p in ("id", "identifier", "key", "ref", "token", "code"))

    @staticmethod
    def _looks_real(value: str) -> bool:
        """Heuristic: does this value look like real data vs. something the LLM fabricated?"""
        if not isinstance(value, str):
            return True
        if len(value) < 2:
            return False
        # Typical LLM fabrications are generic patterns
        fabricated = {"123", "456", "abc", "test", "sample", "example", "12345"}
        return value.lower() not in fabricated
