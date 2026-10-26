"""Repair trigger policy."""

from conversation_factory.evaluation.judge import JudgeScores
from conversation_factory.evaluation.repair_policy import (
    merge_structural_issues,
    repair_issues_unresolved,
    should_attempt_repair,
)


def test_repair_issues_unresolved_detects_persistence():
    before = [
        "The assistant listed Luxury Resort San Diego which did not appear in tool output.",
    ]
    after = [
        "Minor tone issues; Luxury Resort San Diego still not grounded in tools.",
    ]
    assert repair_issues_unresolved(before, after) is True


def test_repair_issues_unresolved_when_fixed():
    before = ["Hotel name invented."]
    after = ["Conversation flows well."]
    assert repair_issues_unresolved(before, after) is False


def test_structural_always_repairs():
    s = JudgeScores(
        naturalness=5.0,
        tool_correctness=5.0,
        task_completion=5.0,
        coherence=5.0,
    )
    assert should_attempt_repair(
        s,
        [],
        structural_redundant=True,
        mean_threshold=3.0,
    )


def test_low_mean_repairs():
    s = JudgeScores(
        naturalness=2.0,
        tool_correctness=2.0,
        task_completion=2.0,
        coherence=2.0,
    )
    assert should_attempt_repair(
        s,
        [],
        structural_redundant=False,
        mean_threshold=3.0,
    )


def test_high_mean_but_tool_issues_repairs():
    """Hallucination pattern: decent mean but tool_correctness down + issues."""
    s = JudgeScores(
        naturalness=5.0,
        tool_correctness=3.4,
        task_completion=5.0,
        coherence=3.4,
    )
    assert should_attempt_repair(
        s,
        ["Invented hotel names"],
        structural_redundant=False,
        mean_threshold=3.0,
    )


def test_high_scores_no_issues_skips():
    s = JudgeScores(
        naturalness=5.0,
        tool_correctness=5.0,
        task_completion=5.0,
        coherence=5.0,
    )
    assert not should_attempt_repair(
        s,
        [],
        structural_redundant=False,
        mean_threshold=3.0,
    )


def test_merge_structural():
    assert "runaway" in " ".join(
        merge_structural_issues(["x"], True)
    ).lower()
