"""Tests for generate() cohort sampling (multi-step + multi-tool vs remainder)."""

import random

import pytest

from conversation_factory.pipeline import sampling_constraints_for_cohort


@pytest.mark.parametrize(
    "index,num,ratio,expected_multi,expected_min_steps",
    [
        (0, 10, 0.55, True, 3),  # max(3, 2)
        (4, 10, 0.55, True, 3),
        (5, 10, 0.55, False, 2),
        (0, 4, 0.5, True, 3),
        (2, 4, 0.5, False, 2),
    ],
)
def test_cohort_flags(index, num, ratio, expected_multi, expected_min_steps):
    rng = random.Random(42)
    c = sampling_constraints_for_cohort(
        index,
        num,
        multi_step_ratio=ratio,
        short_conversation_ratio=0.0,
        min_steps=2,
        max_steps=6,
        min_multi_step_calls=3,
        strict_sampler_walk=True,
        rng=rng,
    )
    assert c.require_multi_tool is expected_multi
    assert c.min_steps == expected_min_steps
    assert c.min_distinct_tools == (2 if expected_multi else 1)


def test_parallel_chain_count_sets_pattern():
    c0 = sampling_constraints_for_cohort(
        0,
        10,
        multi_step_ratio=0.6,
        short_conversation_ratio=0.0,
        min_steps=2,
        max_steps=6,
        min_multi_step_calls=3,
        strict_sampler_walk=True,
        rng=random.Random(0),
        parallel_chain_count=2,
    )
    assert c0.pattern == "parallel"
    c1 = sampling_constraints_for_cohort(
        1,
        10,
        multi_step_ratio=0.6,
        short_conversation_ratio=0.0,
        min_steps=2,
        max_steps=6,
        min_multi_step_calls=3,
        strict_sampler_walk=True,
        rng=random.Random(0),
        parallel_chain_count=2,
    )
    assert c1.pattern == "parallel"


def test_multi_cohort_respects_higher_cli_min_steps():
    """If --min-steps is 5, multi cohort must be at least 5 endpoints."""
    rng = random.Random(0)
    c = sampling_constraints_for_cohort(
        0,
        5,
        multi_step_ratio=1.0,
        short_conversation_ratio=0.0,
        min_steps=5,
        max_steps=8,
        min_multi_step_calls=3,
        strict_sampler_walk=True,
        rng=rng,
    )
    assert c.min_steps == 5
    assert c.require_multi_tool is True


def test_short_cohort_first_indices():
    c = sampling_constraints_for_cohort(
        0,
        10,
        multi_step_ratio=0.55,
        short_conversation_ratio=0.2,
        min_steps=2,
        max_steps=6,
        min_multi_step_calls=3,
        strict_sampler_walk=True,
        rng=random.Random(0),
    )
    assert c.min_steps == 1
    assert c.max_steps == 2
    assert c.require_multi_tool is False
