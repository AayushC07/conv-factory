"""Generic classification of ToolBench-style FQNs ``ToolName/operation_name``.

Works from operation names and path segments alone — no per-API allowlists — so it
scales to large heterogeneous catalogs (thousands of tools).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolRegistry


def endpoint_operation_name(fqn: str) -> str:
    """Return the operation segment (after the first ``/``), or the whole string."""
    if not fqn or "/" not in fqn:
        return (fqn or "").strip()
    return fqn.split("/", 1)[1].strip()


def is_likely_listing_search_operation(fqn: str) -> bool:
    """True if the operation name suggests listing, search, or discovery (not a single-entity GET)."""
    op = endpoint_operation_name(fqn).lower()
    if not op:
        return False
    # Common ToolBench / OpenAPI style: search_*, query_*, find_*, …
    for prefix in ("search", "query", "find", "browse", "discover", "lookup"):
        if op.startswith(prefix):
            return True
    # Single-resource reads: get_x, fetch_x — unless they embed search/list semantics
    if op.startswith(("get_", "fetch_")) and not any(
        x in op for x in ("search", "list", "query", "find", "browse")
    ):
        return False
    # Broad patterns used across REST and ToolBench naming
    patterns = (
        r"\bsearch\b",
        r"\bquery\b",
        r"\bfind\b",
        r"\bbrowse\b",
        r"\blookup\b",
        r"\bdiscover\b",
        r"\blist\b",
        r"^list_",
        r"_list$",
        r"\ball_\b",
    )
    return any(re.search(p, op) for p in patterns)


def is_likely_commit_operation(fqn: str) -> bool:
    """True if the operation likely mutates state (book, pay, create, etc.) — not a read-only GET."""
    op = endpoint_operation_name(fqn).lower()
    if not op:
        return False
    # Read / introspection prefixes (common across APIs)
    if op.startswith(
        (
            "get_",
            "fetch_",
            "retrieve_",
            "search_",
            "query_",
            "list_",
            "head_",
            "describe_",
            "validate_",
            "check_",
            "preview_",
            "parse_",
        )
    ):
        return False
    if op.startswith(("post_", "put_", "patch_")):
        return True
    # Mutation / side-effect verbs (substring — catches book_flight, make_reservation, etc.)
    return bool(
        re.search(
            r"(^|_)(book|purchase|reserve|checkout|subscribe|order|pay|submit|create|update|delete|cancel|confirm|place|make|add|remove)(_|$)",
            op,
        )
    )


def resolve_endpoint_fqn(raw: str, registry: ToolRegistry | None = None) -> str:
    """Return a canonical FQN when possible; otherwise return *raw* unchanged.

    With a registry: prefer exact match; try a single common ToolBench-style ``Tool_API`` fix
    only when that candidate is registered. Never invent names with no registry confirmation.
    """
    raw = (raw or "").strip()
    if not raw:
        return raw
    if registry is None:
        return raw
    if registry.get_endpoint(raw):
        return raw
    if "/" not in raw:
        return raw
    tool, rest = raw.split("/", 1)
    if "_API" not in tool:
        candidate = f"{tool}_API/{rest}"
        if registry.get_endpoint(candidate):
            return candidate
    return raw
