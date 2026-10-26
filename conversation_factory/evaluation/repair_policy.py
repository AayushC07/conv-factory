"""When to run repair/regenerate after judging (see DESIGN §7)."""

from __future__ import annotations

from typing import Any

from .judge import JudgeScores


def should_attempt_repair(
    scores: JudgeScores,
    issues: list[str] | None,
    *,
    structural_redundant: bool,
    mean_threshold: float,
    per_dimension_floor: float = 3.5,
) -> bool:
    """Return True if we should invoke ConversationRepairer.

    Triggers:
    1. Structural validation failed (e.g. runaway identical tool calls).
    2. Mean judge score below ``mean_threshold`` (default quality gate).
    3. Judge reported issues **and** tool_correctness or coherence is below
       ``per_dimension_floor`` — catches high mean with factual/coherence defects
       (e.g. hallucinated listings) that still pass the mean.

    ``per_dimension_floor`` is set to **3.5** (not 4.5) so minor judge nitpicks alone
    do not trigger repair; see DESIGN §7.
    """
    if structural_redundant:
        return True
    if scores.mean_score < mean_threshold:
        return True
    issue_list = [x for x in (issues or []) if isinstance(x, str) and x.strip()]
    if issue_list and (
        scores.tool_correctness < per_dimension_floor
        or scores.coherence < per_dimension_floor
    ):
        return True
    return False


def repair_issues_unresolved(before: list[str], after: list[str]) -> bool:
    """True if post-repair judge issues still contain the same failure as before.

    Used to reject LLM repairs that re-score better on mean but leave factual
    issues (e.g. hallucinated hotel names) unchanged.
    """
    if not before or not after:
        return False
    blob = " ".join(after).lower()
    for b in before:
        s = (b or "").strip().lower()
        words = s.split()
        if len(words) < 5:
            continue
        # 4-word windows from the substantive part of the issue (skip "the assistant")
        for i in range(2, max(3, len(words) - 3)):
            phrase = " ".join(words[i : i + 4])
            if len(phrase) >= 12 and phrase in blob:
                return True
    return False


def merge_structural_issues(
    issues: list[str] | None, structural_redundant: bool
) -> list[str]:
    if not structural_redundant:
        return list(issues or [])
    extra = (
        "Structural: identical tool+arguments repeated 3+ times consecutively "
        "(runaway loop)."
    )
    return list(dict.fromkeys(list(issues or []) + [extra]))
