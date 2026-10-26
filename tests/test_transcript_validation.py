"""Transcript replay and directions grounding checks."""

from conversation_factory.evaluation.transcript_validation import (
    replay_validate_conversation_messages,
    validate_directions_grounding,
    validate_repaired_transcript,
)


def test_replay_catches_invalid_restaurant_id():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "endpoint": "Restaurant_API/make_reservation",
                    "arguments": {
                        "restaurant_id": "The Cozy Fork",
                        "date": "2023-12-15",
                        "time": "19:00",
                        "party_size": 2,
                    },
                }
            ],
        },
        {"role": "tool", "content": {"status": "success"}},
    ]
    errs = replay_validate_conversation_messages(messages)
    assert errs


def test_directions_grounding_detects_paraphrase():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"endpoint": "Maps_API/get_directions", "arguments": {}}]},
        {
            "role": "tool",
            "content": {
                "data": {
                    "steps": [
                        {"instruction": "Head northwest on N Michigan Ave toward E Ontario St", "distance": "0.2 miles"}
                    ]
                }
            },
        },
        {
            "role": "assistant",
            "content": "1. Head southwest on Magnificent Mile toward Wacker.\n2. Take I-94.",
        },
    ]
    errs = validate_directions_grounding(messages)
    assert errs


def test_directions_grounding_ok_when_verbatim():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"endpoint": "Maps_API/get_directions", "arguments": {}}]},
        {
            "role": "tool",
            "content": {
                "steps": [
                    {"instruction": "Head north on S State St toward E Washington St.", "distance": "0.5 miles"}
                ]
            },
        },
        {
            "role": "assistant",
            "content": "Steps:\n1. Head north on S State St toward E Washington St. (0.5 miles)\n",
        },
    ]
    assert not validate_directions_grounding(messages)


def test_validate_repaired_transcript_combines():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"endpoint": "Maps_API/get_directions", "arguments": {}}]},
        {"role": "tool", "content": {"steps": [{"instruction": "Turn left on A St", "distance": "1mi"}]}},
        {"role": "assistant", "content": "Turn left on A St then continue."},
    ]
    assert not validate_repaired_transcript(messages)
