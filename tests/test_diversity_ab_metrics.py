"""Empirical Run A vs Run B diversity metrics on the fixture corpus (mock LLM).

Reproduces the assignment's diversity experiment with the same seed and
``--no-cross-conversation-steering`` vs default steering.

Set ``WRITE_DIVERSITY_REPORT=1`` to write ``output/diversity_ab_fixture_report.json``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from conversation_factory.evaluation.metrics import corpus_quality_summary
from conversation_factory.pipeline import ConversationPipeline
from tests.fixtures import create_sample_toolbench_dir
from tests.test_e2e import _create_mock_llm


def test_diversity_run_a_vs_b_fixture_metrics():
    """Steering on should not reduce category coverage vs steering off on fixtures."""
    data_dir = create_sample_toolbench_dir()
    mock_llm = _create_mock_llm()
    n = 6
    seed = 42

    async def _generate(steering: bool) -> tuple[list[dict], object]:
        out = tempfile.mkdtemp()
        pipeline = ConversationPipeline(
            data_dir=data_dir,
            output_dir=out,
            seed=seed,
            quality_threshold=2.0,
            cross_conversation_steering=steering,
        )
        pipeline.build()
        pipeline.llm = mock_llm
        pipeline.judge.llm = mock_llm
        pipeline.repairer.llm = mock_llm
        convs = await pipeline.generate(
            num_conversations=n,
            short_conversation_ratio=0.0,
            multi_step_ratio=0.6,
        )
        return convs, pipeline

    conv_a, pipe_a = asyncio.run(_generate(False))
    conv_b, pipe_b = asyncio.run(_generate(True))

    assert len(conv_a) == n
    assert len(conv_b) == n

    cats = pipe_a.registry.categories if pipe_a.registry else []
    endpoints = list(pipe_a.graph.graph.nodes()) if pipe_a.graph else []

    metrics_a = corpus_quality_summary(conv_a, cats, endpoints)
    metrics_b = corpus_quality_summary(conv_b, cats, endpoints)

    report = {
        "seed": seed,
        "num_conversations_per_run": n,
        "run_a_steering_off": metrics_a,
        "run_b_steering_on": metrics_b,
    }

    if os.environ.get("WRITE_DIVERSITY_REPORT") == "1":
        root = Path(__file__).resolve().parents[1] / "output"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "diversity_ab_fixture_report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    cov_a = metrics_a["diversity"]["category_coverage"]
    cov_b = metrics_b["diversity"]["category_coverage"]
    assert cov_b >= cov_a * 0.95, (
        f"expected steering to preserve or improve coverage: A={cov_a} B={cov_b}"
    )

    ent_a = metrics_a["diversity"]["endpoint_entropy"]
    ent_b = metrics_b["diversity"]["endpoint_entropy"]
    assert ent_b > 0 and ent_a > 0
