"""LLM-as-judge for conversation quality evaluation.

Scores conversations on four dimensions:
1. Naturalness — conversational flow and language quality
2. Tool Correctness — validity of tool calls, argument types, response usage
3. Task Completion — whether the user's goal is achieved
4. Coherence — logical chain consistency, proper output referencing

These dimensions were chosen because they directly measure the properties
most important for training tool-use agents: the model needs to learn
when to call tools (coherence), how to call them correctly (tool_correctness),
how to communicate results (naturalness), and how to accomplish goals
(task_completion).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..utils.llm import LLMClient
from .prompts_judge import JUDGE_SYSTEM, JUDGE_PROMPT

logger = logging.getLogger(__name__)


class JudgeScores(BaseModel):
    """Quality scores for a conversation."""

    naturalness: float = Field(ge=1.0, le=5.0, default=3.0)
    tool_correctness: float = Field(ge=1.0, le=5.0, default=3.0)
    task_completion: float = Field(ge=1.0, le=5.0, default=3.0)
    coherence: float = Field(ge=1.0, le=5.0, default=3.0)

    @property
    def mean_score(self) -> float:
        return (
            self.naturalness + self.tool_correctness + self.task_completion + self.coherence
        ) / 4.0

    @property
    def min_score(self) -> float:
        return min(self.naturalness, self.tool_correctness, self.task_completion, self.coherence)

    def to_dict(self) -> dict[str, float]:
        return {
            "naturalness": self.naturalness,
            "tool_correctness": self.tool_correctness,
            "task_completion": self.task_completion,
            "coherence": self.coherence,
        }

    def low_dimensions(self, threshold: float = 3.0) -> list[str]:
        """Return dimension names that scored below threshold."""
        low = []
        for dim, score in self.to_dict().items():
            if score < threshold:
                low.append(dim)
        return low


class LLMJudge:
    """Evaluates conversation quality using an LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        quality_threshold: float = 3.0,
    ) -> None:
        self.llm = llm_client
        self.quality_threshold = quality_threshold

    async def score(self, conversation: dict[str, Any]) -> tuple[JudgeScores, list[str]]:
        """Score a conversation and return (scores, issues)."""
        conv_text = self._format_conversation(conversation)

        prompt = JUDGE_PROMPT.format(conversation=conv_text)

        result = await self.llm.complete_structured(
            prompt,
            system_prompt=JUDGE_SYSTEM,
            temperature=0.1,
            max_tokens=4096,
        )

        scores = self._parse_scores(result)
        issues = result.get("issues", [])

        return scores, issues

    async def score_batch(
        self, conversations: list[dict[str, Any]]
    ) -> list[tuple[JudgeScores, list[str]]]:
        """Score multiple conversations."""
        results = []
        for conv in conversations:
            result = await self.score(conv)
            results.append(result)
        return results

    def passes_threshold(self, scores: JudgeScores) -> bool:
        """Check if scores meet the quality threshold."""
        return scores.mean_score >= self.quality_threshold

    def _format_conversation(self, conversation: dict[str, Any]) -> str:
        messages = conversation.get("messages", [])
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content")
            if msg.get("tool_calls"):
                calls = msg["tool_calls"]
                for tc in calls:
                    ep = tc.get("endpoint", "unknown")
                    args = json.dumps(tc.get("arguments", {}))
                    lines.append(f"ASSISTANT [tool_call]: {ep}({args})")
            elif role == "tool":
                content_str = json.dumps(content, default=str) if isinstance(content, dict) else str(content)
                if len(content_str) > 500:
                    content_str = content_str[:500] + "..."
                lines.append(f"TOOL: {content_str}")
            else:
                lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_scores(result: dict[str, Any]) -> JudgeScores:
        scores_data = result.get("scores", {})
        try:
            return JudgeScores(
                naturalness=float(scores_data.get("naturalness", 3.0)),
                tool_correctness=float(scores_data.get("tool_correctness", 3.0)),
                task_completion=float(scores_data.get("task_completion", 3.0)),
                coherence=float(scores_data.get("coherence", 3.0)),
            )
        except (ValueError, TypeError):
            logger.warning("Failed to parse judge scores, using defaults")
            return JudgeScores()
