"""Unit tests for context management (grounding and steering)."""

import pytest

from conversation_factory.context.grounding import ConversationGrounding
from conversation_factory.context.steering import DiversitySteering
from conversation_factory.execution.session import SessionState
from conversation_factory.graph.sampler import SamplingConstraints, ToolChain


class TestConversationGrounding:
    def test_ground_placeholder(self):
        grounding = ConversationGrounding()
        session = SessionState()
        session.record_call(
            "Hotels_API/search_hotels",
            {"city": "Paris"},
            {"results": [{"id": "htl_001", "name": "Hotel A"}]},
        )

        args = {"hotel_id": "{hotel_id}"}
        grounded = grounding.ground_arguments("Hotels_API/book_hotel", args, session)
        # The placeholder should be replaced with a real value
        assert grounded["hotel_id"] != "{hotel_id}"

    def test_ground_non_placeholder(self):
        grounding = ConversationGrounding()
        session = SessionState()
        args = {"city": "London"}
        grounded = grounding.ground_arguments("test", args, session)
        assert grounded["city"] == "London"

    def test_assistant_instruction_empty(self):
        grounding = ConversationGrounding()
        session = SessionState()
        chain = ToolChain(endpoints=["a/ep1"], pattern="sequential")
        instruction = grounding.get_assistant_instruction(session, chain, 0)
        assert isinstance(instruction, str)

    def test_assistant_instruction_with_history(self):
        grounding = ConversationGrounding()
        session = SessionState()
        session.record_call("a/ep1", {}, {"id": "abc123"})
        chain = ToolChain(endpoints=["a/ep1", "b/ep2"], pattern="sequential")
        instruction = grounding.get_assistant_instruction(session, chain, 1)
        assert "abc123" in instruction or "IMPORTANT" in instruction


class TestDiversitySteering:
    def test_disabled_returns_base(self):
        steering = DiversitySteering(enabled=False)
        base = SamplingConstraints(min_steps=3)
        result = steering.get_sampling_constraints(["A", "B"], base)
        assert result.min_steps == 3

    def test_record_and_track(self):
        steering = DiversitySteering(enabled=True)
        conv = {
            "metadata": {
                "tools_used": ["a/ep1", "b/ep2"],
                "categories": ["Travel"],
                "tool_chain_pattern": "sequential",
            }
        }
        steering.record_conversation(conv)
        stats = steering.get_stats()
        assert stats["total_conversations"] == 1
        assert stats["unique_endpoints_used"] == 2

    def test_steers_toward_underrepresented(self):
        steering = DiversitySteering(enabled=True)

        # Generate 10 "Travel" conversations
        for i in range(10):
            steering.record_conversation({
                "metadata": {
                    "tools_used": [f"hotel/ep{i}"],
                    "categories": ["Travel"],
                    "tool_chain_pattern": "sequential",
                }
            })

        constraints = steering.get_sampling_constraints(
            ["Travel", "Food", "Weather"]
        )
        # Should bias toward Food or Weather (underrepresented)
        if constraints.required_categories:
            assert "Travel" not in constraints.required_categories

    def test_overused_endpoints_excluded(self):
        steering = DiversitySteering(enabled=True)
        for _ in range(20):
            steering.record_conversation({
                "metadata": {
                    "tools_used": ["popular/ep1"],
                    "categories": ["A"],
                    "tool_chain_pattern": "sequential",
                }
            })
        constraints = steering.get_sampling_constraints(["A", "B"])
        assert "popular/ep1" in constraints.excluded_endpoints

    def test_reset(self):
        steering = DiversitySteering(enabled=True)
        steering.record_conversation({
            "metadata": {"tools_used": ["a/b"], "categories": ["X"], "tool_chain_pattern": "seq"}
        })
        steering.reset()
        assert steering.get_stats()["total_conversations"] == 0
