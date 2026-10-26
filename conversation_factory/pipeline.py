"""Main pipeline — ties all components together for end-to-end generation."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from .context.steering import DiversitySteering
from .evaluation.judge import JudgeScores, LLMJudge
from .evaluation.conversation_schema import normalize_conversation
from .evaluation.repair import ConversationRepairer
from .evaluation.conversation_validation import has_redundant_tool_runaway
from .evaluation.metrics import corpus_quality_summary
from .evaluation.repair_policy import merge_structural_issues, should_attempt_repair
from .execution.mock_executor import MockExecutor
from .execution.session import SessionState
from .graph.sampler import SamplingConstraints, ToolChain, ToolChainSampler
from .graph.tool_graph import ToolGraph
from .agents.orchestrator import ConversationOrchestrator
from .registry.loader import ToolBenchLoader
from .registry.registry import ToolRegistry
from .utils.llm import LLMClient

logger = logging.getLogger(__name__)


def sampling_constraints_for_cohort(
    cohort_index: int,
    num_conversations: int,
    *,
    multi_step_ratio: float,
    short_conversation_ratio: float = 0.0,
    min_steps: int = 1,
    max_steps: int,
    min_multi_step_calls: int,
    strict_sampler_walk: bool,
    rng: random.Random,
    parallel_chain_count: int = 0,
) -> SamplingConstraints:
    """Build sampler constraints for conversation index ``cohort_index``.

    - **Short cohort** (first ``floor(n * short_conversation_ratio)``): 1–2 steps,
      single-tool — yields genuinely short dialogues.
    - **Multi cohort** (next ``floor(n * multi_step_ratio)`` slots among remaining):
      multi-tool, longer chains.
    - **Remainder**: single-tool, ``min_steps``–``max_steps``.
    """
    n_short = int(num_conversations * short_conversation_ratio)
    if cohort_index < n_short:
        return SamplingConstraints(
            min_steps=1,
            max_steps=min(2, max_steps),
            require_multi_tool=False,
            min_distinct_tools=1,
            pattern="sequential",
            strict_graph_walk=strict_sampler_walk,
        )

    idx = cohort_index - n_short
    n_remain = num_conversations - n_short
    n_multi = min(int(num_conversations * multi_step_ratio), n_remain)
    is_multi = idx < n_multi
    # Multi cohort: at least min_multi_step_calls steps.
    # Remainder cohort: at least 2 steps so they aren't trivially single-turn.
    chain_min = max(min_multi_step_calls, min_steps) if is_multi else max(2, min_steps)

    if is_multi:
        if idx < parallel_chain_count:
            pattern = "parallel"
        else:
            pattern = rng.choice(["sequential", "mixed"])
    else:
        pattern = "sequential"

    return SamplingConstraints(
        min_steps=chain_min,
        max_steps=max_steps,
        require_multi_tool=is_multi,
        min_distinct_tools=2 if is_multi else 1,
        pattern=pattern,
        strict_graph_walk=strict_sampler_walk,
    )


class ConversationPipeline:
    """End-to-end pipeline for generating tool-use conversations."""

    def __init__(
        self,
        data_dir: str | Path,
        output_dir: str | Path = "output",
        model: str = "gpt-4o-mini",
        seed: int = 42,
        max_tools: int | None = None,
        quality_threshold: float = 3.0,
        max_repair_attempts: int = 3,
        use_llm_mock: bool = True,
        cross_conversation_steering: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.max_tools = max_tools
        self.quality_threshold = quality_threshold
        self.max_repair_attempts = max_repair_attempts
        self.use_llm_mock = use_llm_mock
        self._model = model

        self.rng = random.Random(seed)

        self._llm: Optional[LLMClient] = None
        self._judge: Optional[LLMJudge] = None
        self._repairer: Optional[ConversationRepairer] = None
        self.registry: Optional[ToolRegistry] = None
        self.graph: Optional[ToolGraph] = None
        self.sampler: Optional[ToolChainSampler] = None
        self.steering = DiversitySteering(enabled=cross_conversation_steering)

    @property
    def llm(self) -> LLMClient:
        """Lazily initialize LLM client (requires OPENAI_API_KEY)."""
        if self._llm is None:
            self._llm = LLMClient(model=self._model)
        return self._llm

    @llm.setter
    def llm(self, value: LLMClient) -> None:
        """Allow tests to inject a mock LLM."""
        self._llm = value
        # Reset dependent components so they use the new LLM
        self._judge = None
        self._repairer = None

    @property
    def judge(self) -> LLMJudge:
        """Lazily initialize LLM-based judge."""
        if self._judge is None:
            self._judge = LLMJudge(self.llm, quality_threshold=self.quality_threshold)
        return self._judge

    @judge.setter
    def judge(self, value: LLMJudge) -> None:
        """Allow tests to inject a mock judge."""
        self._judge = value

    @property
    def repairer(self) -> ConversationRepairer:
        """Lazily initialize conversation repairer."""
        if self._repairer is None:
            self._repairer = ConversationRepairer(
                self.llm,
                self.judge,
                max_repair_attempts=self.max_repair_attempts,
                quality_threshold=self.quality_threshold,
                registry=self.registry,
            )
        return self._repairer

    @repairer.setter
    def repairer(self, value: ConversationRepairer) -> None:
        """Allow tests to inject a mock repairer."""
        self._repairer = value

    def build(self) -> dict[str, Any]:
        """Build artifacts: registry, graph, indexes."""
        logger.info(f"Building artifacts from {self.data_dir}...")

        # Load tools
        loader = ToolBenchLoader(self.data_dir)
        tools = loader.load_all(max_tools=self.max_tools)

        # Build registry
        self.registry = ToolRegistry()
        self.registry.add_tools(tools)
        registry_path = self.output_dir / "registry.json"
        self.registry.save(registry_path)

        # Build graph
        self.graph = ToolGraph()
        self.graph.build(self.registry)
        graph_path = self.output_dir / "tool_graph.json"
        self.graph.save(graph_path)

        # Create sampler
        self.sampler = ToolChainSampler(self.graph, seed=self.seed)

        summary = {
            "registry": self.registry.summary(),
            "graph": self.graph.summary(),
            "artifacts": {
                "registry": str(registry_path),
                "graph": str(graph_path),
            },
        }
        logger.info(f"Build complete: {summary}")

        # Save build summary
        with open(self.output_dir / "build_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def load_artifacts(self) -> None:
        """Load previously built artifacts."""
        registry_path = self.output_dir / "registry.json"
        graph_path = self.output_dir / "tool_graph.json"

        if registry_path.exists():
            self.registry = ToolRegistry.load(registry_path)
        if graph_path.exists():
            self.graph = ToolGraph.load(graph_path)

        if self.registry and self.graph:
            self.sampler = ToolChainSampler(self.graph, seed=self.seed)

    async def generate(
        self,
        num_conversations: int = 10,
        min_steps: int = 1,
        max_steps: int = 6,
        multi_step_ratio: float = 0.65,
        short_conversation_ratio: float = 0.15,
        min_multi_step_calls: int = 3,
        strict_sampler_walk: bool = True,
        parallel_chain_count: int = 2,
        concurrency: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate a corpus of conversations.

        Conversations are generated in batches of ``concurrency`` for
        throughput.  Steering is updated between batches so diversity
        constraints remain effective.
        """
        if not self.registry or not self.graph or not self.sampler:
            self.load_artifacts()
            if not self.registry or not self.graph or not self.sampler:
                raise RuntimeError("Artifacts not built. Run 'build' first.")

        executor = MockExecutor(
            self.registry, self.llm, use_llm=self.use_llm_mock, seed=self.seed
        )

        conversations: list[dict[str, Any]] = []
        num_failed = 0
        num_fallback_samples = 0
        available_categories = self.registry.categories
        all_endpoints = list(self.graph.graph.nodes())
        endpoint_category = {
            n: self.graph.get_node_data(n).get("category", "Other")
            for n in self.graph.graph.nodes()
        }

        self.steering.set_planned_corpus_size(num_conversations)

        # Process in batches for parallelism while preserving steering accuracy
        indices = list(range(num_conversations))
        pbar = tqdm(total=num_conversations, desc="Generating conversations")

        for batch_start in range(0, num_conversations, concurrency):
            batch_indices = indices[batch_start : batch_start + concurrency]

            # Pre-sample chains with current steering state (must be sequential)
            batch_plans: list[tuple[int, str, ToolChain]] = []
            for i in batch_indices:
                base_constraints = sampling_constraints_for_cohort(
                    i,
                    num_conversations,
                    multi_step_ratio=multi_step_ratio,
                    short_conversation_ratio=short_conversation_ratio,
                    min_steps=min_steps,
                    max_steps=max_steps,
                    min_multi_step_calls=min_multi_step_calls,
                    strict_sampler_walk=strict_sampler_walk,
                    rng=self.rng,
                    parallel_chain_count=parallel_chain_count,
                )
                constraints = self.steering.get_sampling_constraints(
                    available_categories,
                    base_constraints,
                    all_endpoints=all_endpoints,
                    endpoint_category=endpoint_category,
                )
                tool_chain = self.sampler.sample(constraints)
                batch_plans.append((i, f"conv_{i:04d}", tool_chain))

            # Generate batch in parallel
            tasks = [
                self._generate_single(
                    idx, conv_id, tool_chain, executor, available_categories,
                    all_endpoints, endpoint_category,
                )
                for idx, conv_id, tool_chain in batch_plans
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch conversation failed: {result}")
                    num_failed += 1
                    pbar.update(1)
                    continue
                if result is None:
                    num_failed += 1
                    pbar.update(1)
                    continue
                if result.get("metadata", {}).get("fallback_sample"):
                    num_fallback_samples += 1
                conversations.append(result)
                self.steering.record_conversation(result)
                pbar.update(1)

        pbar.close()

        # Save dataset
        output_path = self.output_dir / "dataset.jsonl"
        self._save_dataset(conversations, output_path)

        # Compute and save corpus metrics
        metrics = corpus_quality_summary(
            conversations, available_categories, all_endpoints
        )
        metrics["steering_stats"] = self.steering.get_stats()
        metrics["generation_stats"] = {
            "requested": num_conversations,
            "generated": len(conversations),
            "failed": num_failed,
            "attrition_rate": round(num_failed / max(num_conversations, 1), 3),
            "fallback_samples": num_fallback_samples,
        }
        if num_failed > 0:
            logger.warning(
                f"Generation attrition: {num_failed}/{num_conversations} conversations "
                f"failed ({num_failed / num_conversations:.1%})"
            )
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Generated {len(conversations)} conversations -> {output_path}")
        return conversations

    async def _generate_single(
        self,
        idx: int,
        conv_id: str,
        tool_chain: ToolChain,
        executor: MockExecutor,
        available_categories: list[str],
        all_endpoints: list[str],
        endpoint_category: dict[str, str],
    ) -> dict[str, Any] | None:
        """Generate, judge, and optionally repair a single conversation."""
        # Each coroutine gets its own orchestrator to avoid shared state
        orchestrator = ConversationOrchestrator(
            self.registry, executor, self.llm, max_turns=15
        )
        try:
            conversation = await orchestrator.generate_conversation(
                tool_chain, conversation_id=conv_id, seed=self.seed + idx
            )

            # Retry once if conversation has 0 tool calls (LLM didn't engage with tools)
            meta = conversation.get("metadata", {})
            if meta.get("num_tool_calls", 0) == 0 and len(tool_chain.endpoints) > 0:
                logger.warning(
                    f"Conversation {conv_id} had 0 tool calls "
                    f"(chain has {len(tool_chain.endpoints)} endpoints), retrying..."
                )
                conversation = await orchestrator.generate_conversation(
                    tool_chain, conversation_id=conv_id, seed=self.seed + idx + 500
                )

            # Propagate sampler fallback flag to conversation metadata
            if tool_chain.metadata.get("fallback"):
                conversation.setdefault("metadata", {})["fallback_sample"] = True

            # Judge quality
            scores, issues = await self.judge.score(conversation)
            conversation["judge_scores"] = scores.to_dict()

            structural_redundant = has_redundant_tool_runaway(
                conversation.get("messages", [])
            ) or conversation.get("metadata", {}).get(
                "aborted_redundant_identical_tools", False
            )
            issues = merge_structural_issues(issues, structural_redundant)
            conversation["judge_issues"] = issues

            needs_repair = should_attempt_repair(
                scores,
                issues,
                structural_redundant=structural_redundant,
                mean_threshold=self.quality_threshold,
            )

            if needs_repair:
                why = (
                    "structural redundant tools"
                    if structural_redundant
                    else (
                        "low tool_correctness/coherence with issues"
                        if issues
                        and (
                            scores.tool_correctness < 4.5
                            or scores.coherence < 4.5
                        )
                        else f"score {scores.mean_score:.2f}"
                    )
                )
                logger.info(
                    f"Conversation {conv_id} ({why}), attempting repair..."
                )

                async def regenerate():
                    return await orchestrator.generate_conversation(
                        tool_chain, conversation_id=conv_id, seed=self.seed + idx + 1000
                    )

                # Build tool descriptions so the repair LLM knows what tools are available
                tool_desc_parts = []
                for ep_fqn in tool_chain.endpoints:
                    result = self.registry.get_endpoint(ep_fqn) if self.registry else None
                    if result:
                        tool, ep = result
                        req = [p.name for p in ep.required_parameters]
                        opt = [p.name for p in ep.optional_parameters]
                        tool_desc_parts.append(
                            f"- {ep_fqn}: {ep.description or 'No description'} "
                            f"(required: {req}, optional: {opt})"
                        )
                tool_descriptions = "\n".join(tool_desc_parts) if tool_desc_parts else ""

                conversation, scores, issues, action = await self.repairer.repair_or_regenerate(
                    conversation, scores, issues,
                    regenerate_fn=regenerate,
                    tool_descriptions=tool_descriptions,
                )
                conversation["judge_scores"] = scores.to_dict()
                conversation["judge_issues"] = issues
                conversation["metadata"]["repair_action"] = action
                conversation["metadata"]["repair_reason"] = [why] + list(issues or [])

            normalize_conversation(conversation, self.registry)
            return conversation

        except Exception as e:
            logger.error(f"Failed to generate conversation {conv_id}: {e}")
            return None

    async def evaluate(
        self, dataset_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Evaluate a dataset and compute metrics."""
        path = Path(dataset_path) if dataset_path else self.output_dir / "dataset.jsonl"
        conversations = self._load_dataset(path)

        if not conversations:
            raise ValueError(f"No conversations found in {path}")

        # Re-score if needed
        scored = []
        for conv in tqdm(conversations, desc="Evaluating"):
            if "judge_scores" not in conv or not conv["judge_scores"]:
                scores, issues = await self.judge.score(conv)
                conv["judge_scores"] = scores.to_dict()
                conv["judge_issues"] = issues
            scored.append(conv)

        # Compute metrics
        categories = self.registry.categories if self.registry else []
        if not self.graph:
            self.load_artifacts()
        all_endpoints = list(self.graph.graph.nodes()) if self.graph else []
        metrics = corpus_quality_summary(scored, categories, all_endpoints)

        # Save
        self._save_dataset(scored, path)
        eval_path = self.output_dir / "evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Evaluation complete: {metrics}")
        return metrics

    def _save_dataset(
        self, conversations: list[dict[str, Any]], path: Path
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for conv in conversations:
                # Normalize is idempotent — only needed for evaluate() path
                # where conversations may not have gone through _generate_single.
                normalize_conversation(conv, self.registry)
                f.write(json.dumps(conv, default=str) + "\n")

    @staticmethod
    def _load_dataset(path: Path) -> list[dict[str, Any]]:
        conversations = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        conversations.append(json.loads(line))
        return conversations
