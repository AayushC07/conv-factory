"""conversation_schema normalization and metadata sync."""

from conversation_factory.evaluation.conversation_schema import (
    ensure_closing_assistant_after_tool_tail,
    normalize_conversation,
    normalize_messages,
    resolve_endpoint_fqn,
    sanitize_messages_for_export,
    sync_metadata_from_messages,
    validate_messages_schema,
)
from conversation_factory.registry.loader import ToolBenchLoader
from conversation_factory.registry.registry import ToolRegistry
from tests.fixtures import create_sample_toolbench_dir


def _sample_registry() -> ToolRegistry:
    loader = ToolBenchLoader(create_sample_toolbench_dir())
    tools = loader.load_all(max_tools=6)
    reg = ToolRegistry()
    reg.add_tools(tools)
    return reg


def test_resolve_endpoint_fqn_registry_confirms_api_suffix():
    reg = _sample_registry()
    # LLM typo: short tool name → canonical only if registered
    fixed = resolve_endpoint_fqn("Events_API/search_events", reg)
    assert fixed == "Events_API/search_events"


def test_resolve_endpoint_fqn_without_registry_is_passthrough():
    assert resolve_endpoint_fqn("SomeTool/do_thing", None) == "SomeTool/do_thing"


def test_normalize_messages_maps_tool_and_args():
    raw = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"tool": "Events_API/search_events", "args": {"city": "NYC"}},
            ],
        }
    ]
    reg = _sample_registry()
    out = normalize_messages(raw, reg)
    assert out[0]["tool_calls"][0] == {
        "endpoint": "Events_API/search_events",
        "arguments": {"city": "NYC"},
    }


def test_normalize_messages_drops_empty_tool_calls_key():
    raw = [
        {"role": "assistant", "content": "Hello?", "tool_calls": []},
    ]
    out = normalize_messages(raw)
    assert "tool_calls" not in out[0]


def test_normalize_messages_drops_assistant_prose_when_tool_calls():
    raw = [
        {
            "role": "assistant",
            "content": "Let me search for you.",
            "tool_calls": [{"endpoint": "Events_API/search_events", "arguments": {"city": "NYC"}}],
        }
    ]
    reg = _sample_registry()
    out = normalize_messages(raw, reg)
    assert out[0]["content"] is None


def test_sanitize_messages_fixes_assistant_key_typo():
    raw = [
        {"role": "user", "content": "hi"},
        {"ASSISTANT": "I found a restaurant.", "extra": 1},
    ]
    out, warnings = sanitize_messages_for_export(raw)
    assert len(warnings) >= 1
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == "I found a restaurant."
    assert validate_messages_schema(out) == []


def test_validate_messages_schema_rejects_bad_role():
    raw = [{"role": "narrator", "content": "oops"}]
    assert len(validate_messages_schema(raw)) == 1


def test_ensure_closing_assistant_after_tool_tail():
    raw = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{"endpoint": "W/x", "arguments": {}}]},
        {"role": "tool", "content": {"ok": True}},
    ]
    out, w = ensure_closing_assistant_after_tool_tail(raw)
    assert len(w) == 1
    assert out[-1]["role"] == "assistant"
    assert out[-1].get("content")
    assert validate_messages_schema(out) == []


def test_normalize_conversation_sanitizes_assistant_typo():
    conv = {
        "conversation_id": "c1",
        "messages": [{"role": "user", "content": "x"}, {"ASSISTANT": "Hello"}],
        "metadata": {},
    }
    normalize_conversation(conv, None)
    assert conv["messages"][1]["role"] == "assistant"
    assert conv["messages"][1]["content"] == "Hello"
    assert conv["metadata"].get("schema_fixes")


def test_sanitize_corrects_tool_call_role():
    """LLM-hallucinated 'tool_call' role should be corrected to 'assistant'."""
    raw = [
        {"role": "user", "content": "hi"},
        {"role": "tool_call", "content": None, "tool_calls": [{"endpoint": "X/y", "arguments": {}}]},
    ]
    out, warnings = sanitize_messages_for_export(raw)
    assert out[1]["role"] == "assistant"
    assert any("corrected hallucinated role" in w for w in warnings)


def test_sanitize_corrects_tool_calls_role():
    """LLM-hallucinated 'tool_calls' role should be corrected to 'assistant'."""
    raw = [
        {"role": "tool_calls", "content": None, "tool_calls": [{"endpoint": "X/y", "arguments": {}}]},
    ]
    out, warnings = sanitize_messages_for_export(raw)
    assert out[0]["role"] == "assistant"


def test_sanitize_corrects_function_role():
    """LLM-hallucinated 'function' role should be corrected to 'assistant'."""
    raw = [{"role": "function", "content": "calling tool"}]
    out, warnings = sanitize_messages_for_export(raw)
    assert out[0]["role"] == "assistant"


def test_sanitize_corrects_observation_to_tool():
    """LLM-hallucinated 'observation' role should be corrected to 'tool'."""
    raw = [{"role": "observation", "content": {"result": 42}}]
    out, warnings = sanitize_messages_for_export(raw)
    assert out[0]["role"] == "tool"


def test_sanitize_drops_unknown_role():
    """Completely unknown roles should be dropped, not passed through."""
    raw = [
        {"role": "user", "content": "hi"},
        {"role": "narrator", "content": "this should not exist"},
        {"role": "assistant", "content": "hello"},
    ]
    out, warnings = sanitize_messages_for_export(raw)
    assert len(out) == 2
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"
    assert any("dropped message with unrecognized role" in w for w in warnings)


def test_normalize_conversation_fixes_hallucinated_roles_end_to_end():
    """Full pipeline: hallucinated roles are corrected and no schema errors remain."""
    conv = {
        "conversation_id": "c_fix",
        "messages": [
            {"role": "user", "content": "book me a hotel"},
            {"role": "tool_calls", "content": None, "tool_calls": [{"endpoint": "H/search", "arguments": {"city": "NYC"}}]},
            {"role": "tool_response", "content": {"results": []}},
            {"role": "assistant", "content": "I found some options."},
        ],
        "metadata": {},
    }
    normalize_conversation(conv, None)
    roles = [m["role"] for m in conv["messages"]]
    assert all(r in ("user", "assistant", "tool", "system") for r in roles)
    assert roles == ["user", "assistant", "tool", "assistant"]
    # Should have schema_fixes for the corrected roles
    assert conv["metadata"].get("schema_fixes")


def test_sync_metadata_from_messages():
    conv = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"endpoint": "Hotels_API/search_hotels", "arguments": {"q": "x"}},
                    {"endpoint": "Hotels_API/search_hotels", "arguments": {"q": "y"}},
                ],
            },
            {"role": "tool", "content": {}},
            {"role": "tool", "content": {}},
        ],
        "metadata": {"seed": 1},
    }
    sync_metadata_from_messages(conv)
    assert conv["metadata"]["tools_used"] == ["Hotels_API/search_hotels"]
    assert conv["metadata"]["num_tool_calls"] == 2
    assert conv["metadata"]["num_distinct_tools"] == 1
    assert conv["metadata"]["num_turns"] == 4
    assert conv["metadata"]["seed"] == 1
