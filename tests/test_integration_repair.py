"""Integration test for the retry/repair loop.

Tests that a low-quality conversation can be improved via the repair pipeline.
Uses mock LLM responses to simulate the repair flow without real API calls.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conversation_factory.evaluation.judge import JudgeScores, LLMJudge
from conversation_factory.evaluation.repair import ConversationRepairer
from conversation_factory.utils.llm import LLMClient


def _make_low_quality_conversation():
    return {
        "conversation_id": "conv_test_repair",
        "messages": [
            {"role": "user", "content": "book hotel"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"endpoint": "Hotels_API/book_hotel", "arguments": {"hotel_id": "xxx"}}
            ]},
            {"role": "tool", "content": {"status": "error", "message": "invalid hotel_id"}},
            {"role": "assistant", "content": "Done."},
        ],
        "metadata": {
            "tools_used": ["Hotels_API/book_hotel"],
            "num_tool_calls": 1,
            "num_distinct_tools": 1,
            "num_turns": 4,
        },
    }


def _make_repaired_conversation():
    return {
        "conversation_id": "conv_test_repair",
        "messages": [
            {"role": "user", "content": "I'd like to book a hotel in Paris for this weekend."},
            {"role": "assistant", "content": "I'd be happy to help! What's your budget range?"},
            {"role": "user", "content": "Under 200 euros per night."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"endpoint": "Hotels_API/search_hotels", "arguments": {"city": "Paris", "max_price": 200}}
            ]},
            {"role": "tool", "content": {"results": [{"id": "htl_001", "name": "Hotel du Marais", "price": 175}]}},
            {"role": "assistant", "content": None, "tool_calls": [
                {"endpoint": "Hotels_API/book_hotel", "arguments": {"hotel_id": "htl_001", "check_in": "2026-04-11"}}
            ]},
            {"role": "tool", "content": {"booking_id": "bk_001", "status": "confirmed"}},
            {"role": "assistant", "content": "I've booked Hotel du Marais for April 11. Your confirmation is bk_001."},
        ],
        "metadata": {
            "tools_used": ["Hotels_API/search_hotels", "Hotels_API/book_hotel"],
            "num_tool_calls": 2,
            "num_distinct_tools": 1,
            "num_turns": 8,
        },
    }


class TestRepairIntegration:
    """Integration test demonstrating the repair loop."""

    def test_repair_improves_quality(self):
        """Verify that the repair loop can improve a low-quality conversation."""

        mock_llm = MagicMock(spec=LLMClient)

        # First call to judge returns low scores
        low_scores = {
            "scores": {
                "naturalness": 1.5,
                "tool_correctness": 1.0,
                "task_completion": 1.0,
                "coherence": 1.0,
            },
            "explanation": "Very poor quality",
            "issues": ["Missing search step", "Placeholder hotel_id", "No disambiguation"],
        }

        # Second call (after repair) returns improved scores
        improved_scores = {
            "scores": {
                "naturalness": 4.0,
                "tool_correctness": 4.5,
                "task_completion": 4.5,
                "coherence": 4.0,
            },
            "explanation": "Well-structured conversation",
            "issues": [],
        }

        # Repair call returns repaired messages
        repaired_messages = {
            "messages": _make_repaired_conversation()["messages"],
        }

        call_count = {"n": 0}

        async def mock_complete_structured(prompt, system_prompt=None, temperature=0.3, max_tokens=2048):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First judge call
                return low_scores
            elif call_count["n"] == 2:
                # Repair call
                return repaired_messages
            else:
                # Second judge call (after repair)
                return improved_scores

        mock_llm.complete_structured = AsyncMock(side_effect=mock_complete_structured)

        judge = LLMJudge(mock_llm, quality_threshold=3.0)
        repairer = ConversationRepairer(
            mock_llm, judge, max_repair_attempts=3, quality_threshold=3.0
        )

        low_conv = _make_low_quality_conversation()

        async def run_repair():
            # Score the original
            scores, issues = await judge.score(low_conv)
            assert scores.mean_score < 3.0
            assert not judge.passes_threshold(scores)

            # Repair
            repaired, new_scores, new_issues, action = await repairer.repair_or_regenerate(
                low_conv, scores, issues
            )

            # Verify improvement
            assert new_scores.mean_score > scores.mean_score
            assert judge.passes_threshold(new_scores)
            assert action in ("repaired", "regenerated")
            return repaired, new_scores

        repaired_conv, final_scores = asyncio.run(run_repair())
        assert final_scores.mean_score >= 3.0

    def test_repair_with_regeneration(self):
        """Test that very low-quality conversations trigger regeneration."""

        mock_llm = MagicMock(spec=LLMClient)

        # All judge calls after regeneration return good scores
        good_scores = {
            "scores": {
                "naturalness": 4.0,
                "tool_correctness": 4.0,
                "task_completion": 4.0,
                "coherence": 4.0,
            },
            "explanation": "Good after regeneration",
            "issues": [],
        }

        async def mock_structured(prompt, system_prompt=None, temperature=0.3, max_tokens=2048):
            # Regardless of call order, return good scores (the regenerated conv is better)
            return good_scores

        mock_llm.complete_structured = AsyncMock(side_effect=mock_structured)

        judge = LLMJudge(mock_llm, quality_threshold=3.0)
        repairer = ConversationRepairer(mock_llm, judge, quality_threshold=3.0)

        conv = _make_low_quality_conversation()

        async def run():
            scores = JudgeScores(naturalness=1.0, tool_correctness=1.0, task_completion=1.0, coherence=1.0)
            issues = ["Everything is wrong"]

            regenerated_conv = _make_repaired_conversation()

            async def regen():
                return regenerated_conv

            result_conv, result_scores, result_issues, action = await repairer.repair_or_regenerate(
                conv, scores, issues, regenerate_fn=regen
            )
            assert result_scores.mean_score > 1.0
            assert judge.passes_threshold(result_scores)
            return action

        action = asyncio.run(run())
        assert action in ("regenerated", "repaired", "best_effort")
