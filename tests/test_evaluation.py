"""Unit tests for evaluation metrics."""

import math

import pytest

from conversation_factory.evaluation.metrics import (
    category_coverage,
    category_entropy,
    corpus_quality_summary,
    endpoint_entropy,
    mean_pairwise_jaccard_distance,
)
from conversation_factory.evaluation.judge import JudgeScores


class TestJudgeScores:
    def test_mean_score(self):
        scores = JudgeScores(
            naturalness=4.0, tool_correctness=3.0,
            task_completion=5.0, coherence=4.0
        )
        assert scores.mean_score == 4.0

    def test_min_score(self):
        scores = JudgeScores(
            naturalness=4.0, tool_correctness=2.0,
            task_completion=5.0, coherence=4.0
        )
        assert scores.min_score == 2.0

    def test_low_dimensions(self):
        scores = JudgeScores(
            naturalness=4.0, tool_correctness=2.0,
            task_completion=5.0, coherence=2.5
        )
        low = scores.low_dimensions(threshold=3.0)
        assert "tool_correctness" in low
        assert "coherence" in low
        assert "naturalness" not in low


class TestEndpointEntropy:
    def test_uniform_distribution(self):
        convs = [
            {"metadata": {"tools_used": ["a"]}},
            {"metadata": {"tools_used": ["b"]}},
            {"metadata": {"tools_used": ["c"]}},
            {"metadata": {"tools_used": ["d"]}},
        ]
        entropy = endpoint_entropy(convs)
        assert entropy == pytest.approx(2.0, abs=0.01)  # log2(4) = 2

    def test_single_tool(self):
        convs = [
            {"metadata": {"tools_used": ["a"]}},
            {"metadata": {"tools_used": ["a"]}},
        ]
        entropy = endpoint_entropy(convs)
        assert entropy == 0.0

    def test_empty(self):
        assert endpoint_entropy([]) == 0.0


class TestCategoryCoverage:
    def test_full_coverage(self):
        convs = [
            {"metadata": {"categories": ["A", "B"]}},
            {"metadata": {"categories": ["C"]}},
        ]
        assert category_coverage(convs, ["A", "B", "C"]) == 1.0

    def test_partial_coverage(self):
        convs = [
            {"metadata": {"categories": ["A"]}},
        ]
        assert category_coverage(convs, ["A", "B", "C"]) == pytest.approx(1 / 3)

    def test_no_categories(self):
        assert category_coverage([], []) == 0.0


class TestJaccardDistance:
    def test_no_overlap(self):
        convs = [
            {"metadata": {"tools_used": ["a", "b"]}},
            {"metadata": {"tools_used": ["c", "d"]}},
        ]
        dist = mean_pairwise_jaccard_distance(convs)
        assert dist == 1.0

    def test_identical_tools(self):
        convs = [
            {"metadata": {"tools_used": ["a", "b"]}},
            {"metadata": {"tools_used": ["a", "b"]}},
        ]
        dist = mean_pairwise_jaccard_distance(convs)
        assert dist == 0.0

    def test_partial_overlap(self):
        convs = [
            {"metadata": {"tools_used": ["a", "b"]}},
            {"metadata": {"tools_used": ["b", "c"]}},
        ]
        dist = mean_pairwise_jaccard_distance(convs)
        # Intersection = {b}, Union = {a, b, c} => J = 1/3 => dist = 2/3
        assert dist == pytest.approx(2 / 3, abs=0.01)


class TestCorpusQualitySummary:
    def test_basic_summary(self):
        convs = [
            {
                "judge_scores": {"naturalness": 4.0, "tool_correctness": 4.0,
                                  "task_completion": 4.0, "coherence": 4.0},
                "metadata": {
                    "tools_used": ["a/ep1", "b/ep2"],
                    "categories": ["Travel"],
                    "num_tool_calls": 3,
                    "num_distinct_tools": 2,
                    "num_turns": 7,
                },
            },
            {
                "judge_scores": {"naturalness": 3.0, "tool_correctness": 3.0,
                                  "task_completion": 3.0, "coherence": 3.0},
                "metadata": {
                    "tools_used": ["c/ep3"],
                    "categories": ["Food"],
                    "num_tool_calls": 1,
                    "num_distinct_tools": 1,
                    "num_turns": 4,
                },
            },
        ]
        summary = corpus_quality_summary(
            convs,
            ["Travel", "Food", "Weather"],
            ["a/ep1", "b/ep2", "c/ep3"],
        )
        assert summary["num_conversations"] == 2
        assert summary["quality"]["mean_judge_score"] == 3.5
        assert summary["coverage"]["multi_step_ratio"] == 0.5
        assert summary["diversity"]["category_coverage"] == pytest.approx(2 / 3)
        assert "category_entropy" in summary["diversity"]
        assert summary["diversity"]["endpoint_entropy_normalized"] >= 0.0
        assert summary["diversity"]["category_entropy_max_bits"] > 0.0


class TestCategoryEntropy:
    def test_skewed_categories(self):
        convs = [
            {"metadata": {"categories": ["Travel", "Travel"]}},
            {"metadata": {"categories": ["Travel"]}},
        ]
        h = category_entropy(convs, ["Travel", "Food", "Weather", "X", "Y"])
        assert h < math.log2(5)  # not uniform over 5
