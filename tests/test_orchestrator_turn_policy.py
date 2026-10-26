"""Orchestrator turn policy (booking after search)."""

from conversation_factory.agents.assistant_agent import AssistantAction
from conversation_factory.agents.orchestrator_turn_policy import (
    action_includes_booking,
    user_message_after_last_listing_search,
)


def test_user_message_after_search_required():
    msgs = [
        {"role": "user", "content": "find food"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"endpoint": "Restaurant_API/search_restaurants", "arguments": {"location": "x"}},
        ]},
        {"role": "tool", "content": {"results": []}},
    ]
    assert user_message_after_last_listing_search(msgs) is False


def test_user_message_after_search_ok():
    msgs = [
        {"role": "user", "content": "find food"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"endpoint": "Restaurant_API/search_restaurants", "arguments": {"location": "x"}},
        ]},
        {"role": "tool", "content": {"results": []}},
        {"role": "user", "content": "I'll take the first one"},
    ]
    assert user_message_after_last_listing_search(msgs) is True


def test_parallel_search_tools_then_user():
    msgs = [
        {"role": "user", "content": "trip"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"endpoint": "Hotels_API/search_hotels", "arguments": {}},
            {"endpoint": "Flights_API/search_flights", "arguments": {}},
        ]},
        {"role": "tool", "content": {}},
        {"role": "tool", "content": {}},
        {"role": "user", "content": "book the morning flight"},
    ]
    assert user_message_after_last_listing_search(msgs) is True


def test_action_includes_booking():
    a = AssistantAction(
        "call_tool",
        tool_calls=[{"endpoint": "Hotels_API/book_hotel", "arguments": {"hotel_id": "1"}}],
    )
    assert action_includes_booking(a) is True
