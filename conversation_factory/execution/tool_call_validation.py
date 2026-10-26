"""Pre-execution checks for tool calls (domain mixups, placeholders, duplicate list calls)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..registry.endpoint_semantics import (
    endpoint_operation_name,
    is_likely_commit_operation,
    is_likely_listing_search_operation,
)
from .session import SessionState

if TYPE_CHECKING:
    from ..registry.registry import ToolRegistry

# Literal placeholders — do not call location-based search with these; ask the user instead.
_INVALID_LOCATION_TOKENS = frozenset(
    {
        "",
        "unknown",
        "n/a",
        "na",
        "none",
        "null",
        "tbd",
        "any",
        "...",
        "???",
        "somewhere",
        "anywhere",
    }
)

# Common parameter names for geospatial / keyword search across OpenAPI-style tools
_LOCATION_LIKE_KEYS = frozenset(
    {
        "city",
        "location",
        "origin",
        "destination",
        "address",
        "query",
        "q",
        "keyword",
        "region",
        "country",
        "from",
        "to",
        "near",
    }
)


def _norm_loc(val: Any) -> str | None:
    if not isinstance(val, str):
        return None
    s = val.strip().lower()
    return s if s else None


def validate_search_location(
    endpoint_fqn: str, arguments: dict[str, Any]
) -> dict[str, Any] | None:
    """Block listing-search calls when location-like arguments are placeholders."""
    if not is_likely_listing_search_operation(endpoint_fqn):
        return None

    checks: list[tuple[str, Any]] = []
    op = endpoint_operation_name(endpoint_fqn).lower()
    # Flight-style routes often use origin/destination
    if "flight" in op or "route" in op:
        for key in ("origin", "destination", "from", "to"):
            if key in arguments:
                checks.append((key, arguments.get(key)))
    for key, val in arguments.items():
        if key in _LOCATION_LIKE_KEYS and (key, val) not in checks:
            checks.append((key, val))

    for key, raw in checks:
        n = _norm_loc(raw)
        if n is None:
            continue
        if n in _INVALID_LOCATION_TOKENS:
            return {
                "status": "error",
                "error": (
                    f"Invalid {key!r}: ask the user for a real value before searching; "
                    "do not use placeholder tokens."
                ),
            }
    return None


def _looks_like_free_text_name(val: str) -> bool:
    """Heuristic: multi-word title without digits — not a typical API id."""
    v = val.strip()
    if len(v) < 6:
        return False
    if any(c.isdigit() for c in v):
        return False
    return len(v.split()) >= 2


def validate_commit_id_arguments(
    endpoint_fqn: str, arguments: dict[str, Any]
) -> dict[str, Any] | None:
    """Reject *id fields that look like free-text names for likely commit operations."""
    if not is_likely_commit_operation(endpoint_fqn):
        return None
    for key, val in arguments.items():
        lk = key.lower()
        if not (lk == "id" or lk.endswith("_id")):
            continue
        if not isinstance(val, str) or not val.strip():
            continue
        v = val.strip()
        if " " in v:
            return {
                "status": "error",
                "error": (
                    f"{key!r} must be an identifier copied from a prior search/list result, "
                    "not a display name or sentence."
                ),
            }
        if _looks_like_free_text_name(v):
            return {
                "status": "error",
                "error": (
                    f"{key!r} looks like a place name, not an id from a search/list tool "
                    "(use the id field returned by the matching search endpoint)."
                ),
            }
    return None


def _walk_collect_ids(
    obj: Any, out: set[str], *, keys: frozenset[str] = frozenset({"id", "restaurant_id", "hotel_id"})
) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()
            if lk in keys and isinstance(v, str) and v.strip():
                out.add(v.strip())
            _walk_collect_ids(v, out, keys=keys)
    elif isinstance(obj, list):
        for x in obj:
            _walk_collect_ids(x, out, keys=keys)


def collect_restaurant_ids_from_session(session: SessionState) -> set[str]:
    ids: set[str] = set()
    for entry in session.history:
        ep = entry.get("endpoint") or ""
        if "Restaurant" not in ep:
            continue
        _walk_collect_ids(entry.get("response"), ids, keys=frozenset({"id", "restaurant_id"}))
    return ids


def collect_hotel_ids_from_search_hotels(session: SessionState) -> set[str]:
    """Hotel ids that appear in Hotels_API/search_hotels responses (booking must use these)."""
    ids: set[str] = set()
    for entry in session.history:
        ep = (entry.get("endpoint") or "").lower()
        if "search_hotels" not in ep:
            continue
        _walk_collect_ids(entry.get("response"), ids, keys=frozenset({"id", "hotel_id"}))
    return ids


def validate_restaurant_reservation_id(
    endpoint_fqn: str, arguments: dict[str, Any], session: SessionState
) -> dict[str, Any] | None:
    """Block make_reservation unless restaurant_id comes from Restaurant search results."""
    if "make_reservation" not in endpoint_fqn.lower():
        return None
    rid = arguments.get("restaurant_id") or arguments.get("id")
    if not isinstance(rid, str) or not rid.strip():
        return None
    known = collect_restaurant_ids_from_session(session)
    if not known:
        return {
            "status": "error",
            "error": (
                "Call Restaurant_API/search_restaurants or get_restaurant_details first and use a "
                "restaurant id from its results for make_reservation; do not use a place name or "
                "Maps search alone."
            ),
        }
    if rid.strip() not in known:
        return {
            "status": "error",
            "error": (
                "restaurant_id must be copied from a Restaurant_API response in this session. "
                "If you only have a place name from Maps search, call Restaurant_API/search_restaurants first."
            ),
        }
    return None


def validate_hotel_booking_id(
    endpoint_fqn: str, arguments: dict[str, Any], session: SessionState
) -> dict[str, Any] | None:
    """Block book_hotel unless hotel_id came from Hotels_API/search_hotels in this session."""
    if "book_hotel" not in endpoint_fqn.lower():
        return None
    hid = arguments.get("hotel_id") or arguments.get("id")
    if not isinstance(hid, str) or not hid.strip():
        return None
    known = collect_hotel_ids_from_search_hotels(session)
    if not known:
        return {
            "status": "error",
            "error": (
                "Call Hotels_API/search_hotels first and use a hotel id from its results for book_hotel; "
                "do not use airport codes or city codes as hotel_id."
            ),
        }
    if hid.strip() not in known:
        return {
            "status": "error",
            "error": (
                "hotel_id must be copied from a Hotels_API/search_hotels response in this session."
            ),
        }
    return None


def validate_duplicate_search(
    endpoint_fqn: str, arguments: dict[str, Any], session: SessionState
) -> dict[str, Any] | None:
    """Block a second identical listing-search call with the same arguments."""
    if not is_likely_listing_search_operation(endpoint_fqn):
        return None
    sig = json.dumps(arguments, sort_keys=True, default=str)
    for entry in session.history:
        if entry.get("endpoint") != endpoint_fqn:
            continue
        prev = entry.get("arguments") or {}
        if json.dumps(prev, sort_keys=True, default=str) == sig:
            return {
                "status": "error",
                "error": (
                    "Duplicate search with identical parameters; results were already returned above. "
                    "Use those results or change filters."
                ),
            }
    return None


def _walk_collect_flight_ids(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()
            if lk in ("flight_id", "flightid") and isinstance(v, str):
                out.add(v)
            if lk == "data" and isinstance(v, dict):
                flights = v.get("flights")
                if isinstance(flights, list):
                    for f in flights:
                        if isinstance(f, dict) and isinstance(f.get("flight_id"), str):
                            out.add(f["flight_id"])
            if lk == "results" and isinstance(v, list):
                for row in v:
                    if isinstance(row, dict) and isinstance(row.get("flight_id"), str):
                        out.add(row["flight_id"])
            _walk_collect_flight_ids(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _walk_collect_flight_ids(x, out)


def _walk_collect_event_ids(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "event_id" and isinstance(v, str):
                out.add(v)
        if "events" in obj and isinstance(obj["events"], list):
            for ev in obj["events"]:
                if isinstance(ev, dict):
                    for ik in ("id", "event_id"):
                        val = ev.get(ik)
                        if isinstance(val, str):
                            out.add(val)
        if "results" in obj and isinstance(obj["results"], list):
            for row in obj["results"]:
                if isinstance(row, dict):
                    if isinstance(row.get("event_id"), str):
                        out.add(row["event_id"])
                    if "ticket_price" in row or "venue" in row:
                        rid = row.get("id")
                        if isinstance(rid, str):
                            out.add(rid)
        for v in obj.values():
            _walk_collect_event_ids(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _walk_collect_event_ids(x, out)


def collect_flight_ids_from_session(session: SessionState) -> set[str]:
    ids: set[str] = set()
    for entry in session.history:
        _walk_collect_flight_ids(entry.get("response"), ids)
    return ids


def collect_event_ids_from_session(session: SessionState) -> set[str]:
    ids: set[str] = set()
    for entry in session.history:
        _walk_collect_event_ids(entry.get("response"), ids)
    return ids


def _is_flight_booking_operation(fqn: str) -> bool:
    op = endpoint_operation_name(fqn).lower()
    return "flight" in op and is_likely_commit_operation(fqn)


def validate_registry_required_parameters(
    endpoint_fqn: str,
    arguments: dict[str, Any],
    registry: "ToolRegistry",
) -> dict[str, Any] | None:
    """Block calls when required endpoint parameters are missing or empty."""
    result = registry.get_endpoint(endpoint_fqn)
    if result is None:
        return None
    _tool, ep = result
    for p in ep.required_parameters:
        val = arguments.get(p.name)
        missing = val is None
        if not missing and isinstance(val, str) and not val.strip():
            missing = True
        if not missing and isinstance(val, str):
            n = _norm_loc(val)
            if n is not None and n in _INVALID_LOCATION_TOKENS:
                missing = True
        if missing:
            return {
                "status": "error",
                "error": (
                    f"Missing or empty required parameter {p.name!r} for {endpoint_fqn}. "
                    "Ask the user or use a value from a prior tool result."
                ),
            }
    return None


def validate_before_execute(
    endpoint_fqn: str,
    arguments: dict[str, Any],
    session: SessionState,
    *,
    registry: "ToolRegistry | None" = None,
) -> dict[str, Any] | None:
    """If invalid, return an error payload to use instead of executor output; else None."""

    if registry is not None:
        req_err = validate_registry_required_parameters(
            endpoint_fqn, arguments, registry
        )
        if req_err is not None:
            return req_err

    dup = validate_duplicate_search(endpoint_fqn, arguments, session)
    if dup is not None:
        return dup

    loc = validate_search_location(endpoint_fqn, arguments)
    if loc is not None:
        return loc

    id_err = validate_commit_id_arguments(endpoint_fqn, arguments)
    if id_err is not None:
        return id_err

    res_err = validate_restaurant_reservation_id(endpoint_fqn, arguments, session)
    if res_err is not None:
        return res_err

    hotel_err = validate_hotel_booking_id(endpoint_fqn, arguments, session)
    if hotel_err is not None:
        return hotel_err

    if not _is_flight_booking_operation(endpoint_fqn):
        return None

    fid = arguments.get("flight_id")
    if not isinstance(fid, str) or not fid.strip():
        return None

    flight_ids = collect_flight_ids_from_session(session)
    event_ids = collect_event_ids_from_session(session)

    if event_ids and fid in event_ids and fid not in flight_ids:
        return {
            "status": "error",
            "error": (
                "This flight_id does not appear in any flight-search result; "
                "it matches an id from a different domain (e.g. events). "
                "Use an id returned by the flight search operation for this tool."
            ),
            "flight_id": fid,
        }

    return None
