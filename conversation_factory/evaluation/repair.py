"""Automatic retry/repair for low-quality conversations.

Strategy:
1. If a conversation fails validation (structural issues), regenerate it entirely.
2. If it scores below threshold on specific dimensions, attempt targeted repair:
   - Feed the conversation + identified issues to a repair agent
   - Re-score the repaired version
   - Accept if improved, otherwise try regeneration
3. Maximum 3 repair attempts before accepting best-effort or discarding.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Callable, Awaitable, Optional

from ..registry.registry import ToolRegistry
from ..utils.llm import LLMClient
from .conversation_schema import normalize_conversation
from .judge import JudgeScores, LLMJudge
from .repair_policy import repair_issues_unresolved
from .prompts_judge import REPAIR_SYSTEM, REPAIR_PROMPT
from .transcript_validation import validate_repaired_transcript

logger = logging.getLogger(__name__)


class ConversationRepairer:
    """Attempts to repair low-quality conversations."""

    def __init__(
        self,
        llm_client: LLMClient,
        judge: LLMJudge,
        max_repair_attempts: int = 3,
        quality_threshold: float = 3.0,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.llm = llm_client
        self.judge = judge
        self.max_repair_attempts = max_repair_attempts
        self.quality_threshold = quality_threshold
        self.registry = registry

    async def repair_or_regenerate(
        self,
        conversation: dict[str, Any],
        scores: JudgeScores,
        issues: list[str],
        regenerate_fn: Optional[Callable[[], Awaitable[dict[str, Any]]]] = None,
        tool_descriptions: str = "",
    ) -> tuple[dict[str, Any], JudgeScores, list[str], str]:
        """Attempt to repair a conversation, or regenerate if repair fails.

        Returns: (conversation, scores, issues, action_taken)
        """
        original_issues = list(issues or [])
        best_conv = conversation
        best_scores = scores
        best_issues = issues

        for attempt in range(self.max_repair_attempts):
            logger.info(
                f"Repair attempt {attempt + 1}/{self.max_repair_attempts} "
                f"(current mean: {best_scores.mean_score:.2f})"
            )

            if best_scores.mean_score < 2.0 and regenerate_fn and attempt == 0:
                # Very low quality — try full regeneration first
                try:
                    new_conv = await regenerate_fn()
                    new_scores, new_issues = await self.judge.score(new_conv)
                    if new_scores.mean_score > best_scores.mean_score:
                        best_conv = new_conv
                        best_scores = new_scores
                        best_issues = new_issues
                        if self.judge.passes_threshold(best_scores):
                            return best_conv, best_scores, best_issues, "regenerated"
                except Exception as e:
                    logger.warning(f"Regeneration failed: {e}")

            # Attempt targeted repair
            try:
                repaired = await self._repair(
                    best_conv, best_scores, best_issues, tool_descriptions
                )
                if repaired:
                    new_scores, new_issues = await self.judge.score(repaired)
                    if repair_issues_unresolved(original_issues, new_issues):
                        logger.info(
                            "Repair did not clear overlapping judge issues; keeping prior best."
                        )
                        continue
                    improved = new_scores.mean_score > best_scores.mean_score or len(
                        new_issues
                    ) < len(best_issues)
                    if improved:
                        best_conv = repaired
                        best_scores = new_scores
                        best_issues = new_issues
                        if self.judge.passes_threshold(best_scores):
                            return best_conv, best_scores, best_issues, "repaired"
            except Exception as e:
                logger.warning(f"Repair attempt {attempt + 1} failed: {e}")

        if best_scores.mean_score > scores.mean_score:
            action = "repaired"
        elif repair_issues_unresolved(original_issues, best_issues):
            action = "repair_failed_validation"
        else:
            action = "best_effort"
        return best_conv, best_scores, best_issues, action

    async def _repair(
        self,
        conversation: dict[str, Any],
        scores: JudgeScores,
        issues: list[str],
        tool_descriptions: str,
    ) -> dict[str, Any] | None:
        """Attempt a single repair of the conversation."""
        conv_text = self._format_conversation(conversation)
        low_dims = scores.low_dimensions(self.quality_threshold)

        prompt = REPAIR_PROMPT.format(
            conversation=conv_text,
            issues="\n".join(f"- {issue}" for issue in issues) if issues else "No specific issues identified",
            low_scores=", ".join(f"{d}: {scores.to_dict()[d]:.1f}" for d in low_dims),
            tool_descriptions=tool_descriptions or "(same tools as original)",
        )

        result = await self.llm.complete_structured(
            prompt,
            system_prompt=REPAIR_SYSTEM,
            temperature=0.3,
            max_tokens=16384,
        )

        messages = result.get("messages", [])
        if not messages:
            return None

        repaired = copy.deepcopy(conversation)
        repaired["messages"] = messages
        repaired["metadata"] = repaired.get("metadata", {})
        repaired["metadata"]["repaired"] = True
        repaired["metadata"]["repair_llm_low_dimensions"] = low_dims
        normalize_conversation(repaired, self.registry)

        v_errs = validate_repaired_transcript(
            repaired.get("messages") or [], registry=self.registry
        )
        if v_errs:
            logger.info(
                "Repair transcript failed validation (%d issues); discarding repair. %s",
                len(v_errs),
                v_errs[:3],
            )
            return None

        return repaired

    @staticmethod
    def _format_conversation(conversation: dict[str, Any]) -> str:
        messages = conversation.get("messages", [])
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content")
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    ep = tc.get("endpoint", "?")
                    args = json.dumps(tc.get("arguments", {}))
                    lines.append(f"ASSISTANT [tool_call]: {ep}({args})")
            elif role == "tool":
                content_str = json.dumps(content, default=str) if isinstance(content, dict) else str(content)
                lines.append(f"TOOL: {content_str}")
            else:
                lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)
