"""CLI for the conversation factory pipeline.

Commands:
  build     — Ingest ToolBench data, build registry and tool graph
  sample    — Sample tool chains from the graph (JSON + optional HTML viz)
  mock      — Run schema-only mock executor on a demo tool chain (verify offline execution)
  generate  — Generate conversations from the tool graph
  evaluate  — Score and analyze generated conversations
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from .execution.mock_executor import MockExecutor
from .execution.session import SessionState
from .graph.chain_export import chains_document, save_chains_json, write_chains_html
from .graph.sampler import SamplingConstraints, ToolChainSampler
from .graph.tool_graph import ToolGraph
from .pipeline import ConversationPipeline
from .registry.registry import ToolRegistry


def _load_project_env() -> None:
    """Load `.env` from the repository root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        root = Path(__file__).resolve().parent.parent
        env_path = root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Conversation Factory — Multi-Agent Tool-Use Conversation Generator"""
    _load_project_env()
    _setup_logging(verbose)


@cli.command()
@click.option("--data-dir", "-d", required=True, help="Path to ToolBench data directory")
@click.option("--output-dir", "-o", default="output", help="Output directory for artifacts")
@click.option("--max-tools", "-m", type=int, default=None, help="Maximum number of tools to load")
def build(data_dir: str, output_dir: str, max_tools: int | None) -> None:
    """Ingest ToolBench data and build all derived artifacts."""
    pipeline = ConversationPipeline(data_dir=data_dir, output_dir=output_dir, max_tools=max_tools)
    summary = pipeline.build()
    click.echo(json.dumps(summary, indent=2))
    click.echo(f"\nArtifacts saved to {output_dir}/")


@cli.command()
@click.option("--output-dir", "-o", default="output", help="Directory containing tool_graph.json")
@click.option(
    "--graph",
    "-g",
    "graph_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to tool_graph.json (default: OUTPUT_DIR/tool_graph.json)",
)
@click.option("--num-chains", "-n", type=int, default=10, help="Number of chains to sample")
@click.option("--seed", "-s", type=int, default=42, help="Random seed")
@click.option("--min-steps", type=int, default=2, help="Min endpoints per chain")
@click.option("--max-steps", type=int, default=6, help="Max endpoints per chain")
@click.option(
    "--require-multi-tool",
    is_flag=True,
    help="Require at least two distinct tools in each chain",
)
@click.option(
    "--required-category",
    multiple=True,
    help="Category that must appear at least once (repeat flag for multiple)",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON path (default: OUTPUT_DIR/sampled_tool_chains.json)",
)
@click.option(
    "--no-html",
    is_flag=True,
    help="Do not write sampled_tool_chains.html next to the JSON file",
)
@click.option(
    "--strict/--no-strict",
    "strict_walk",
    default=True,
    help="Strict graph walk: do not randomly jump when stuck (better ordering)",
)
def sample(
    output_dir: str,
    graph_path: Path | None,
    num_chains: int,
    seed: int,
    min_steps: int,
    max_steps: int,
    require_multi_tool: bool,
    required_category: tuple[str, ...],
    out: Path | None,
    no_html: bool,
    strict_walk: bool,
) -> None:
    """Sample tool chains from the built tool graph; save JSON and optional HTML visualization."""
    resolved_graph = graph_path if graph_path is not None else Path(output_dir) / "tool_graph.json"
    if not resolved_graph.is_file():
        raise click.ClickException(
            f"Tool graph not found: {resolved_graph}. Run `convfactory build --data-dir ...` first."
        )

    g = ToolGraph.load(resolved_graph)
    constraints = SamplingConstraints(
        min_steps=min_steps,
        max_steps=max_steps,
        require_multi_tool=require_multi_tool,
        min_distinct_tools=2 if require_multi_tool else 1,
        required_categories=list(required_category),
        strict_graph_walk=strict_walk,
    )
    sampler = ToolChainSampler(g, seed=seed)
    chains = sampler.sample_batch(num_chains, constraints)

    doc = chains_document(
        chains,
        seed=seed,
        constraints=constraints,
        graph_path=str(resolved_graph.resolve()),
    )
    json_path = out if out is not None else Path(output_dir) / "sampled_tool_chains.json"
    save_chains_json(json_path, doc)

    html_path = json_path.with_suffix(".html")
    if not no_html:
        write_chains_html(html_path, doc)

    click.echo(f"Sampled {len(chains)} chains (seed={seed})")
    click.echo(f"JSON: {json_path.resolve()}")
    if not no_html:
        click.echo(f"HTML: {html_path.resolve()}")
    click.echo("")
    for ch in chains:
        click.echo("  " + " → ".join(ch.endpoints))


async def _run_default_mock_demo(
    executor: MockExecutor, session: SessionState, seed: int
) -> dict[str, object]:
    """Hotels: search → get_hotel_details → book_hotel using IDs from prior steps."""
    steps: list[dict[str, object]] = []

    r1 = await executor.execute(
        "Hotels_API/search_hotels",
        {"city": "Paris"},
        session,
    )
    steps.append(
        {
            "endpoint": "Hotels_API/search_hotels",
            "arguments": {"city": "Paris"},
            "response": r1,
        }
    )
    hotel_id = (r1.get("results") or [{}])[0].get("id")
    if not hotel_id:
        raise RuntimeError("Mock search returned no result id — demo cannot continue")

    r2 = await executor.execute(
        "Hotels_API/get_hotel_details",
        {"hotel_id": hotel_id},
        session,
    )
    steps.append(
        {
            "endpoint": "Hotels_API/get_hotel_details",
            "arguments": {"hotel_id": hotel_id},
            "response": r2,
        }
    )

    r3 = await executor.execute(
        "Hotels_API/book_hotel",
        {
            "hotel_id": hotel_id,
            "check_in": "2026-04-11",
            "check_out": "2026-04-13",
            "guest_name": "Demo Traveler",
        },
        session,
    )
    steps.append(
        {
            "endpoint": "Hotels_API/book_hotel",
            "arguments": {
                "hotel_id": hotel_id,
                "check_in": "2026-04-11",
                "check_out": "2026-04-13",
                "guest_name": "Demo Traveler",
            },
            "response": r3,
        }
    )

    return {
        "seed": seed,
        "schema_only": True,
        "steps": steps,
        "session_call_count": len(session.history),
        "hotel_id_threaded": hotel_id,
        "context_tail": session.get_context_summary()[-400:],
    }


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    default="output",
    help="Directory containing registry.json",
)
@click.option("--seed", "-s", type=int, default=42, help="RNG seed for mock responses")
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write trace JSON (default: OUTPUT_DIR/mock_executor_trace.json)",
)
def mock(output_dir: str, seed: int, out: Path | None) -> None:
    """Run schema-only MockExecutor on a fixed Hotels chain (search → details → book).

    Verifies offline execution and session chaining without calling an LLM for mocks.
    """
    reg_path = Path(output_dir) / "registry.json"
    if not reg_path.is_file():
        raise click.ClickException(
            f"Missing {reg_path}. Run: convfactory build --data-dir <tools> --output-dir {output_dir}"
        )

    registry = ToolRegistry.load(reg_path)
    executor = MockExecutor(registry, llm_client=None, use_llm=False, seed=seed)
    session = SessionState()

    trace = asyncio.run(_run_default_mock_demo(executor, session, seed))
    out_path = out if out is not None else Path(output_dir) / "mock_executor_trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)

    click.echo(json.dumps(trace, indent=2, default=str))
    click.echo(f"\nTrace saved to {out_path.resolve()}")


@cli.command()
@click.option("--data-dir", "-d", required=True, help="Path to ToolBench data directory")
@click.option("--output-dir", "-o", default="output", help="Output directory")
@click.option("--num-conversations", "-n", type=int, default=10, help="Number of conversations")
@click.option("--seed", "-s", type=int, default=42, help="Random seed")
@click.option("--model", default="gpt-4o-mini", help="LLM model to use")
@click.option("--max-tools", "-m", type=int, default=None, help="Max tools to load")
@click.option(
    "--min-steps",
    type=int,
    default=1,
    help="Min sampled endpoints for the short cohort (default 1 for short conversations)",
)
@click.option("--max-steps", type=int, default=6, help="Max tool calls per conversation")
@click.option("--quality-threshold", type=float, default=3.0, help="Min quality score (1-5)")
@click.option("--max-repair-attempts", type=int, default=3, help="Max repair attempts")
@click.option("--no-cross-conversation-steering", is_flag=True, help="Disable cross-conversation steering")
@click.option("--no-llm-mock", is_flag=True, help="Use schema-derived mocks instead of LLM")
@click.option(
    "--multi-step-ratio",
    type=float,
    default=0.65,
    show_default=True,
    help="Fraction of corpus that is multi-step multi-tool (>=3 calls, >=2 distinct tools)",
)
@click.option(
    "--short-conversation-ratio",
    type=float,
    default=0.15,
    show_default=True,
    help="Fraction of corpus that is short (1–2 sampled steps, single-tool)",
)
@click.option(
    "--min-multi-step-calls",
    type=int,
    default=3,
    show_default=True,
    help="Minimum sampled endpoints for the multi-step/multi-tool cohort (default 3+)",
)
@click.option(
    "--no-strict-sampler-walk",
    is_flag=True,
    help="Allow random jumps when the walk has no out-edges (weaker endpoint ordering)",
)
@click.option(
    "--parallel-chain-count",
    type=int,
    default=2,
    show_default=True,
    help="First N multi-step conversations use pattern=parallel (rest sequential/mixed)",
)
@click.option(
    "--concurrency",
    type=int,
    default=5,
    show_default=True,
    help="Number of conversations to generate in parallel per batch",
)
def generate(
    data_dir: str,
    output_dir: str,
    num_conversations: int,
    seed: int,
    model: str,
    max_tools: int | None,
    min_steps: int,
    max_steps: int,
    quality_threshold: float,
    max_repair_attempts: int,
    no_cross_conversation_steering: bool,
    no_llm_mock: bool,
    multi_step_ratio: float,
    short_conversation_ratio: float,
    min_multi_step_calls: int,
    no_strict_sampler_walk: bool,
    parallel_chain_count: int,
    concurrency: int,
) -> None:
    """Generate synthetic tool-use conversations."""
    pipeline = ConversationPipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        model=model,
        seed=seed,
        max_tools=max_tools,
        quality_threshold=quality_threshold,
        max_repair_attempts=max_repair_attempts,
        use_llm_mock=not no_llm_mock,
        cross_conversation_steering=not no_cross_conversation_steering,
    )

    # Build first if artifacts don't exist
    if not (pipeline.output_dir / "registry.json").exists():
        click.echo("Building artifacts first...")
        pipeline.build()

    if not 0.0 <= multi_step_ratio <= 1.0:
        raise click.ClickException("--multi-step-ratio must be between 0 and 1")
    if min_multi_step_calls < 2:
        raise click.ClickException("--min-multi-step-calls must be at least 2")
    if parallel_chain_count < 0:
        raise click.ClickException("--parallel-chain-count must be >= 0")
    if not 0.0 <= short_conversation_ratio < 1.0:
        raise click.ClickException("--short-conversation-ratio must be in [0, 1)")

    conversations = asyncio.run(
        pipeline.generate(
            num_conversations=num_conversations,
            min_steps=min_steps,
            max_steps=max_steps,
            multi_step_ratio=multi_step_ratio,
            short_conversation_ratio=short_conversation_ratio,
            min_multi_step_calls=min_multi_step_calls,
            strict_sampler_walk=not no_strict_sampler_walk,
            parallel_chain_count=parallel_chain_count,
            concurrency=concurrency,
        )
    )

    click.echo(f"\nGenerated {len(conversations)} conversations")
    click.echo(f"Dataset saved to {output_dir}/dataset.jsonl")
    click.echo(f"Metrics saved to {output_dir}/metrics.json")


@cli.command()
@click.option("--data-dir", "-d", default="", help="Path to ToolBench data directory (for category info)")
@click.option("--output-dir", "-o", default="output", help="Output directory")
@click.option("--dataset", type=str, default=None, help="Path to dataset JSONL file")
@click.option("--model", default="gpt-4o-mini", help="LLM model for judging")
@click.option("--quality-threshold", type=float, default=3.0, help="Min quality score")
def evaluate(
    data_dir: str,
    output_dir: str,
    dataset: str | None,
    model: str,
    quality_threshold: float,
) -> None:
    """Evaluate generated conversations with LLM-as-judge."""
    pipeline = ConversationPipeline(
        data_dir=data_dir or output_dir,
        output_dir=output_dir,
        model=model,
        quality_threshold=quality_threshold,
    )
    pipeline.load_artifacts()

    metrics = asyncio.run(pipeline.evaluate(dataset_path=dataset))
    click.echo(json.dumps(metrics, indent=2, default=str))
    click.echo(f"\nEvaluation saved to {output_dir}/evaluation.json")


if __name__ == "__main__":
    cli()
