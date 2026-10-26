"""User agent — simulates a human user making requests.

Maintains a persona and goal, provides information when asked,
reacts naturally to results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..utils.llm import LLMClient
from .prompts import USER_AGENT_SYSTEM, USER_AGENT_PROMPT

logger = logging.getLogger(__name__)


class UserAgent:
    """Simulates a user in a tool-use conversation."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self.persona: str = ""
        self.scenario: str = ""
        self.missing_info: list[str] = []
        self._revealed_info: list[str] = []

    def configure(
        self,
        persona: str,
        scenario: str,
        initial_request: str,
        missing_info: list[str] | None = None,
    ) -> None:
        self.persona = persona
        self.scenario = scenario
        self.initial_request = initial_request
        self.missing_info = missing_info or []
        self._revealed_info = []

    def get_initial_message(self) -> str:
        return self.initial_request

    async def respond(
        self, history: list[dict[str, Any]], instruction: str = ""
    ) -> str:
        """Generate the next user message given conversation history."""
        history_text = self._format_history(history)

        unrevealed = [
            info for info in self.missing_info if info not in self._revealed_info
        ]

        if not instruction:
            if unrevealed:
                instruction = (
                    "The assistant asked you something. Provide the requested information "
                    "naturally. You can reveal: " + ", ".join(unrevealed[:2])
                )
            else:
                instruction = "Respond naturally to the assistant's last message."

        prompt = USER_AGENT_PROMPT.format(
            persona=self.persona,
            scenario=self.scenario,
            missing_info=", ".join(unrevealed) if unrevealed else "none",
            history=history_text,
            instruction=instruction,
        )

        response = await self.llm.complete(prompt, system_prompt=USER_AGENT_SYSTEM, temperature=0.8)

        # Track revealed info
        for info in unrevealed:
            key_terms = info.lower().split()
            if any(term in response.lower() for term in key_terms if len(term) > 3):
                self._revealed_info.append(info)

        return response.strip()

    @staticmethod
    def _format_history(history: list[dict[str, Any]]) -> str:
        lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if msg.get("tool_calls"):
                tools = msg["tool_calls"]
                content = f"[Called tools: {', '.join(tc.get('endpoint', '?') for tc in tools)}]"
            if role == "tool":
                content = f"[Tool result: {str(content)[:200]}]"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
