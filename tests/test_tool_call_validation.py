"""Pre-execution tool validation."""

from conversation_factory.execution.session import SessionState
from conversation_factory.execution.tool_call_validation import (
    collect_event_ids_from_session,
    collect_flight_ids_from_session,
    validate_commit_id_arguments,
    validate_duplicate_search,
    validate_search_location,
    validate_before_execute,
)
from conversation_factory.registry.loader import ToolBenchLoader
from conversation_factory.registry.registry import ToolRegistry
from tests.fixtures import create_sample_toolbench_dir


def _record(session: SessionState, endpoint: str, response: dict) -> None:
    session.record_call(endpoint, {}, response)


def test_validate_before_execute_blocks_missing_required_param_when_registry_provided():
    loader = ToolBenchLoader(create_sample_toolbench_dir())
    reg = ToolRegistry()
    reg.add_tools(loader.load_all(max_tools=6))
    session = SessionState()
    err = validate_before_execute(
        "Hotels_API/get_hotel_details",
        {},
        session,
        registry=reg,
    )
    assert err is not None
    assert err.get("status") == "error"
    assert "hotel_id" in (err.get("error") or "")


def test_book_flight_rejects_event_id_when_flights_searched():
    session = SessionState()
    _record(
        session,
        "Flights_API/search_flights",
        {
            "data": {
                "flights": [
                    {"flight_id": "DL456", "price": 400},
                ]
            }
        },
    )
    _record(
        session,
        "Events_API/search_events",
        {"events": [{"id": "E002", "name": "Concert"}]},
    )
    assert "DL456" in collect_flight_ids_from_session(session)
    assert "E002" in collect_event_ids_from_session(session)

    err = validate_before_execute(
        "Flights_API/book_flight",
        {"flight_id": "E002", "passenger_name": "x"},
        session,
    )
    assert err is not None
    assert err.get("status") == "error"


def test_book_flight_allows_flight_id_from_search():
    session = SessionState()
    _record(
        session,
        "Flights_API/search_flights",
        {"results": [{"flight_id": "DL456", "price": 400}]},
    )
    _record(
        session,
        "Events_API/search_events",
        {"events": [{"id": "E002"}]},
    )
    assert validate_before_execute(
        "Flights_API/book_flight",
        {"flight_id": "DL456"},
        session,
    ) is None


def test_make_reservation_rejects_id_not_from_restaurant_search():
    session = SessionState()
    _record(
        session,
        "Restaurant_API/search_restaurants",
        {"data": [{"id": "42", "name": "Trattoria"}]},
    )
    err = validate_before_execute(
        "Restaurant_API/make_reservation",
        {"restaurant_id": "99", "date": "2024-01-01", "time": "19:00", "party_size": 2},
        session,
    )
    assert err is not None
    assert err.get("status") == "error"


def test_make_reservation_allows_matching_id():
    session = SessionState()
    _record(
        session,
        "Restaurant_API/search_restaurants",
        {"data": [{"id": "42", "name": "Trattoria"}]},
    )
    assert (
        validate_before_execute(
            "Restaurant_API/make_reservation",
            {"restaurant_id": "42", "date": "2024-01-01", "time": "19:00", "party_size": 2},
            session,
        )
        is None
    )


def test_make_reservation_requires_prior_restaurant_call():
    session = SessionState()
    err = validate_before_execute(
        "Restaurant_API/make_reservation",
        {"restaurant_id": "any", "date": "2024-01-01", "time": "19:00", "party_size": 2},
        session,
    )
    assert err is not None
    assert err.get("status") == "error"


def test_book_hotel_rejects_without_search_hotels():
    session = SessionState()
    _record(session, "Hotels_API/get_hotel_details", {"hotel": {"hotel_id": "MIA", "name": "X"}})
    err = validate_before_execute(
        "Hotels_API/book_hotel",
        {"hotel_id": "MIA"},
        session,
    )
    assert err is not None


def test_non_book_endpoint_skips():
    session = SessionState()
    _record(
        session,
        "Hotels_API/search_hotels",
        {"data": {"hotels": [{"id": "1", "name": "Inn"}]}},
    )
    assert (
        validate_before_execute(
            "Hotels_API/book_hotel",
            {"hotel_id": "1"},
            session,
        )
        is None
    )


def test_duplicate_search_blocked():
    session = SessionState()
    session.record_call(
        "Events_API/search_events",
        {"city": "NYC", "date": "2025-06-01"},
        {"events": []},
    )
    err = validate_duplicate_search(
        "Events_API/search_events",
        {"city": "NYC", "date": "2025-06-01"},
        session,
    )
    assert err is not None
    assert err.get("status") == "error"


def test_search_events_rejects_unknown_city():
    err = validate_search_location(
        "Events_API/search_events",
        {"city": "unknown", "date_from": "2023-01-01", "date_to": "2023-01-02"},
    )
    assert err is not None
    assert err.get("status") == "error"


def test_commit_rejects_freeform_name_in_id_field():
    err = validate_commit_id_arguments(
        "Hotels_API/book_hotel",
        {"hotel_id": "Hotel Central Park", "check_in": "2024-01-01", "check_out": "2024-01-02"},
    )
    assert err is not None


def test_validate_before_execute_blocks_unknown_city():
    session = SessionState()
    err = validate_before_execute(
        "Events_API/search_events",
        {"city": "unknown"},
        session,
    )
    assert err is not None


def test_duplicate_search_allows_different_args():
    session = SessionState()
    session.record_call(
        "Events_API/search_events",
        {"city": "NYC"},
        {"events": []},
    )
    assert (
        validate_duplicate_search(
            "Events_API/search_events",
            {"city": "Boston"},
            session,
        )
        is None
    )
