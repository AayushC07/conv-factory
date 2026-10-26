"""Structural validation for generated conversations."""

from conversation_factory.evaluation.conversation_validation import (
    has_redundant_tool_runaway,
    max_consecutive_identical_tool_calls,
)


def test_max_consecutive_single_tool_repeat():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {"endpoint": "X/a", "arguments": {"id": "1"}},
            ],
        },
        {"role": "tool", "content": {}},
        {
            "role": "assistant",
            "tool_calls": [
                {"endpoint": "X/a", "arguments": {"id": "1"}},
            ],
        },
        {"role": "tool", "content": {}},
        {
            "role": "assistant",
            "tool_calls": [
                {"endpoint": "X/a", "arguments": {"id": "1"}},
            ],
        },
        {"role": "tool", "content": {}},
    ]
    assert max_consecutive_identical_tool_calls(messages) == 3
    assert has_redundant_tool_runaway(messages, threshold=3) is True
    assert has_redundant_tool_runaway(messages, threshold=4) is False


def test_max_consecutive_resets_on_different_args():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"endpoint": "X/a", "arguments": {"id": "1"}}],
        },
        {
            "role": "assistant",
            "tool_calls": [{"endpoint": "X/a", "arguments": {"id": "2"}}],
        },
    ]
    assert max_consecutive_identical_tool_calls(messages) == 1
