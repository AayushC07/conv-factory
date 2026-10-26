"""End-to-end test for the full pipeline.

This test:
1. Builds artifacts from sample ToolBench data
2. Generates a dataset (using mock LLM responses)
3. Evaluates the dataset
4. Asserts quality thresholds

For CI without API keys, this test uses a mock LLM client.
Set OPENAI_API_KEY to run against real OpenAI models.
"""

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conversation_factory.pipeline import ConversationPipeline
from conversation_factory.evaluation.metrics import corpus_quality_summary
from tests.fixtures import create_sample_toolbench_dir


def _stub_arguments_for_fqn(fqn: str) -> dict[str, Any]:
    """Minimal valid-looking args for MockExecutor, keyed by endpoint name heuristics."""
    _tool, _, ep = fqn.partition("/")
    name = ep.lower()
    if any(x in name for x in ("search", "list", "find")):
        if "hotel" in name:
            return {"city": "Paris", "max_price": 200}
        if "flight" in name:
            return {"origin": "NYC", "destination": "LON", "date": "2026-06-01"}
        if "restaurant" in name:
            return {"location": "Paris"}
        if "place" in name:
            return {"query": "cafe", "location": "Paris"}
        if "event" in name:
            return {"city": "Paris"}
        return {"query": "test", "city": "Paris"}
    if "weather" in name or "forecast" in name:
        return {"city": "Paris"}
    if "direction" in name:
        return {"origin": "A", "destination": "B"}
    if "detail" in name or (name.startswith("get_") and "search" not in name):
        if "hotel" in name:
            return {"hotel_id": "htl_001"}
        if "restaurant" in name:
            return {"restaurant_id": "rst_001"}
        if "flight" in name:
            return {"flight_id": "flt_001"}
        if "event" in name:
            return {"event_id": "evt_001"}
        return {"id": "id_001"}
    if "book" in name or "reservation" in name or "purchase" in name:
        if "hotel" in name:
            return {
                "hotel_id": "htl_001",
                "check_in": "2026-04-11",
                "check_out": "2026-04-13",
                "guest_name": "Test User",
            }
        if "flight" in name:
            return {"flight_id": "flt_001", "passenger_name": "Test User"}
        if "restaurant" in name:
            return {
                "restaurant_id": "rst_001",
                "date": "2026-04-12",
                "time": "19:00",
                "party_size": 2,
            }
        if "ticket" in name or "event" in name:
            return {"event_id": "evt_001", "quantity": 2}
    return {"query": "test"}


def _dynamic_assistant_sequence(fqns: list[str]) -> list[dict[str, Any]]:
    """Clarify → one call per sampled endpoint → respond (matches AssistantAgent chain filter)."""
    seen: set[str] = set()
    unique: list[str] = []
    for f in fqns:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    seq: list[dict[str, Any]] = [
        {
            "action": "ask_clarification",
            "message": "What's your budget range?",
            "tool_calls": [],
            "reasoning": "Clarify before tools",
        },
    ]
    for fqn in unique:
        seq.append(
            {
                "action": "call_tool",
                "message": None,
                "tool_calls": [
                    {"endpoint": fqn, "arguments": _stub_arguments_for_fqn(fqn)},
                ],
                "reasoning": f"Call {fqn}",
            }
        )
    seq.append(
        {
            "action": "respond",
            "message": "I've completed the planned tool steps for this scenario.",
            "tool_calls": [],
            "reasoning": "Done",
        }
    )
    return seq


@pytest.fixture(autouse=True)
def _e2e_relax_listing_booking_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests-only: mock assistant chains search → … → book without a user turn after listings.

    Production policy requires explicit user choice after search results; the scripted
    mock does not insert that user message, so we bypass the gate for deterministic E2E.
    """
    monkeypatch.setattr(
        "conversation_factory.agents.orchestrator.user_message_after_last_listing_search",
        lambda *_a, **_k: True,
    )


def _create_mock_llm():
    """Create a mock LLM that returns plausible structured outputs.

    Uses separate counters for scenario vs assistant turns so each conversation
    starts with a clean assistant sequence (clarify → tools → respond). Assistant
    actions include **three** tool calls before the final response so
    ``sync_metadata_from_messages`` yields ``num_tool_calls >= 3`` (assignment
    multi-step target).
    """
    mock = MagicMock()
    state = {"scenario_i": 0, "assistant_i": 0}

    scenarios = [
        {
            "scenario": "Book a hotel in Paris and check the weather",
            "user_persona": "a business traveler planning a trip",
            "initial_request": "I need to plan a trip to Paris next week",
            "expected_steps": ["search hotels", "check weather", "book hotel"],
            "disambiguation_needed": True,
            "missing_info": ["budget range", "exact dates"],
        },
        {
            "scenario": "Find restaurants and get directions",
            "user_persona": "a tourist exploring a new city",
            "initial_request": "Can you help me find a good place to eat nearby?",
            "expected_steps": ["search restaurants", "get directions"],
            "disambiguation_needed": True,
            "missing_info": ["cuisine preference", "current location"],
        },
        {
            "scenario": "Search flights and book tickets",
            "user_persona": "a frequent flyer",
            "initial_request": "I need a flight from NYC to London",
            "expected_steps": ["search flights", "book flight"],
            "disambiguation_needed": False,
            "missing_info": ["travel date"],
        },
    ]

    # Three tool calls (multi-step), multiple endpoints / tool APIs (multi-tool).
    assistant_actions = [
        {"action": "ask_clarification", "message": "What's your budget range?", "tool_calls": [], "reasoning": "Need budget info"},
        {"action": "call_tool", "message": None, "tool_calls": [
            {"endpoint": "Hotels_API/search_hotels", "arguments": {"city": "Paris", "max_price": 200}},
        ], "reasoning": "Searching for hotels"},
        {"action": "call_tool", "message": None, "tool_calls": [
            {"endpoint": "Hotels_API/get_hotel_details", "arguments": {"hotel_id": "htl_001"}},
        ], "reasoning": "Details before booking"},
        {"action": "call_tool", "message": None, "tool_calls": [
            {"endpoint": "Weather_API/get_current_weather", "arguments": {"city": "Paris"}},
        ], "reasoning": "Weather check for trip"},
        {"action": "call_tool", "message": None, "tool_calls": [
            {
                "endpoint": "Hotels_API/book_hotel",
                "arguments": {
                    "hotel_id": "htl_001",
                    "check_in": "2026-04-11",
                    "check_out": "2026-04-13",
                    "guest_name": "Test User",
                },
            },
        ], "reasoning": "Booking hotel"},
        {"action": "respond", "message": "I've booked your hotel (bk_001) and confirmed Paris will be mild next week.", "tool_calls": [], "reasoning": "Task complete"},
    ]

    judge_scores = {
        "scores": {"naturalness": 4.0, "tool_correctness": 4.0, "task_completion": 4.0, "coherence": 4.0},
        "explanation": "Good quality conversation",
        "issues": [],
    }

    async def mock_complete(prompt, system_prompt=None, temperature=0.7, max_tokens=2048):
        """Return plausible user responses."""
        if "budget" in prompt.lower() or "clarif" in prompt.lower():
            return "My budget is around 200 euros per night."
        if "react" in prompt.lower() or "thank" in prompt.lower():
            return "That sounds great, thank you!"
        return "Yes, please help me with that."

    async def mock_structured(prompt, system_prompt=None, temperature=0.3, max_tokens=2048):
        """Return structured responses depending on context."""
        pl = prompt.lower()

        if "evaluate this tool-use conversation" in pl:
            return judge_scores

        if "scored poorly" in pl or "conversation repair agent" in pl:
            return {"messages": []}

        if "create a realistic scenario" in pl:
            s = scenarios[state["scenario_i"] % len(scenarios)]
            state["scenario_i"] += 1
            state["assistant_i"] = 0
            return s

        # Assistant: emit tool calls only for endpoints listed in this prompt (sampled chain).
        if "available tools:" in pl and "conversation history:" in pl:
            fqns = re.findall(r"^\s{2}([A-Za-z0-9_]+/[^\s:]+):", prompt, re.MULTILINE)
            if fqns:
                dyn = _dynamic_assistant_sequence(fqns)
                idx = state["assistant_i"] % len(dyn)
                state["assistant_i"] += 1
                return dyn[idx]

        idx = state["assistant_i"] % len(assistant_actions)
        state["assistant_i"] += 1
        return assistant_actions[idx]

    async def mock_chat(messages, temperature=0.7, max_tokens=2048, json_mode=False):
        return "I'd be happy to help you with that."

    mock.complete = AsyncMock(side_effect=mock_complete)
    mock.complete_structured = AsyncMock(side_effect=mock_structured)
    mock.complete_chat = AsyncMock(side_effect=mock_chat)
    mock.total_tokens_used = 0

    return mock


class TestEndToEnd:
    """End-to-end tests using mock LLM."""

    def test_build_artifacts(self):
        """Test that build produces registry and graph files."""
        data_dir = create_sample_toolbench_dir()
        output_dir = tempfile.mkdtemp()

        pipeline = ConversationPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=42,
        )
        summary = pipeline.build()

        assert summary["registry"]["num_tools"] == 6
        assert summary["graph"]["num_nodes"] > 0
        assert Path(output_dir, "registry.json").exists()
        assert Path(output_dir, "tool_graph.json").exists()

    def test_build_and_generate_small(self):
        """Test building and generating a small dataset (5 conversations)."""
        data_dir = create_sample_toolbench_dir()
        output_dir = tempfile.mkdtemp()

        pipeline = ConversationPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=42,
            quality_threshold=2.0,  # Low threshold for testing
        )
        pipeline.build()

        # Replace LLM with mock
        mock_llm = _create_mock_llm()
        pipeline.llm = mock_llm
        pipeline.judge.llm = mock_llm
        pipeline.repairer.llm = mock_llm

        conversations = asyncio.run(
            pipeline.generate(num_conversations=5, concurrency=1)
        )

        assert len(conversations) >= 3  # Some might fail, but most should succeed
        assert Path(output_dir, "dataset.jsonl").exists()

        # Verify dataset format
        with open(Path(output_dir, "dataset.jsonl")) as f:
            for line in f:
                conv = json.loads(line)
                assert "conversation_id" in conv
                assert "messages" in conv
                assert "metadata" in conv
                assert len(conv["messages"]) >= 2

    def test_full_pipeline_with_evaluation(self):
        """Full pipeline: build -> generate -> evaluate."""
        data_dir = create_sample_toolbench_dir()
        output_dir = tempfile.mkdtemp()

        pipeline = ConversationPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=42,
            quality_threshold=2.0,
        )
        pipeline.build()

        mock_llm = _create_mock_llm()
        pipeline.llm = mock_llm
        pipeline.judge.llm = mock_llm
        pipeline.repairer.llm = mock_llm

        conversations = asyncio.run(
            pipeline.generate(num_conversations=5, concurrency=1)
        )

        # Evaluate
        metrics = asyncio.run(pipeline.evaluate())
        assert "num_conversations" in metrics
        assert "quality" in metrics
        assert "diversity" in metrics

    def test_deterministic_with_seed(self):
        """Verify that the same seed produces consistent structure."""
        data_dir = create_sample_toolbench_dir()

        for _ in range(2):
            output_dir = tempfile.mkdtemp()
            pipeline = ConversationPipeline(
                data_dir=data_dir, output_dir=output_dir, seed=42
            )
            summary = pipeline.build()
            assert summary["registry"]["num_tools"] == 6

    def test_cross_conversation_steering_flag(self):
        """Verify steering can be enabled/disabled."""
        data_dir = create_sample_toolbench_dir()
        output_dir = tempfile.mkdtemp()

        # With steering disabled
        pipeline_a = ConversationPipeline(
            data_dir=data_dir, output_dir=output_dir,
            seed=42, cross_conversation_steering=False,
        )
        assert not pipeline_a.steering.enabled

        # With steering enabled
        pipeline_b = ConversationPipeline(
            data_dir=data_dir, output_dir=output_dir,
            seed=42, cross_conversation_steering=True,
        )
        assert pipeline_b.steering.enabled


class TestEndToEndLarge:
    """Large-scale e2e test — generates 100+ conversations.

    This test validates:
    - Dataset has >= 100 conversations
    - Mean judge score exceeds threshold (3.0 for mock, justified in DESIGN.md)
    - Multi-step/multi-tool ratio is in target range
    """

    @pytest.mark.skipif(
        os.environ.get("RUN_LARGE_E2E") != "1"
        and os.environ.get("CI") != "true",
        reason="Large e2e: set RUN_LARGE_E2E=1 locally, or run in CI (CI=true)",
    )
    def test_100_conversations(self):
        data_dir = create_sample_toolbench_dir()
        output_dir = tempfile.mkdtemp()

        pipeline = ConversationPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=42,
            quality_threshold=2.5,
        )
        pipeline.build()

        mock_llm = _create_mock_llm()
        pipeline.llm = mock_llm
        pipeline.judge.llm = mock_llm
        pipeline.repairer.llm = mock_llm

        # Serial generation: the mock LLM uses shared counters in ``state``; parallel
        # batches would race and desync assistant_actions from each conversation.
        conversations = asyncio.run(
            pipeline.generate(
                num_conversations=100,
                short_conversation_ratio=0.0,
                multi_step_ratio=0.6,
                concurrency=1,
            )
        )

        # Threshold: 3.0 is our target mean score.
        # Justification: With mock LLM, scores are deterministic at 4.0.
        # With real LLM, we'd expect variation but mean above 3.0 indicates
        # the pipeline produces structurally sound conversations.
        QUALITY_THRESHOLD = 3.0

        assert len(conversations) >= 100, f"Expected >= 100, got {len(conversations)}"

        # Check quality scores
        scores = []
        for conv in conversations:
            judge = conv.get("judge_scores", {})
            if judge:
                mean = sum(judge.values()) / len(judge)
                scores.append(mean)

        if scores:
            mean_score = sum(scores) / len(scores)
            assert mean_score >= QUALITY_THRESHOLD, (
                f"Mean judge score {mean_score:.2f} below threshold {QUALITY_THRESHOLD}"
            )

        # Check coverage metrics
        all_eps = list(pipeline.graph.graph.nodes()) if pipeline.graph else []
        metrics = corpus_quality_summary(
            conversations, pipeline.registry.categories, all_eps
        )
        multi_step_ratio = metrics["coverage"]["multi_step_ratio"]
        multi_tool_ratio = metrics["coverage"]["multi_tool_ratio"]
        joint_ratio = metrics["coverage"]["joint_multi_step_and_multi_tool_ratio"]

        # Assignment target: ~50–60% of the corpus with multi-step (≥3 calls) and
        # multi-tool (≥2 distinct endpoints in metadata). ``generate`` uses
        # ``multi_step_ratio=0.6`` for the multi cohort; assert realized corpus clears a 50% floor.
        MIN_COHORT_RATIO = 0.5
        assert multi_step_ratio >= MIN_COHORT_RATIO, (
            f"multi_step_ratio {multi_step_ratio:.2f} below {MIN_COHORT_RATIO} "
            "(expected ≥3 tool calls in at least half of conversations)"
        )
        assert multi_tool_ratio >= MIN_COHORT_RATIO, (
            f"multi_tool_ratio {multi_tool_ratio:.2f} below {MIN_COHORT_RATIO} "
            "(expected ≥2 distinct endpoints in at least half of conversations)"
        )
        assert joint_ratio >= MIN_COHORT_RATIO, (
            f"joint multi-step+multi-tool ratio {joint_ratio:.2f} below {MIN_COHORT_RATIO}"
        )
        mean_tool_calls = metrics["coverage"]["mean_tool_calls"]
        assert mean_tool_calls >= 1.0, (
            f"Expected some tool usage in corpus, mean_tool_calls={mean_tool_calls:.2f}"
        )
