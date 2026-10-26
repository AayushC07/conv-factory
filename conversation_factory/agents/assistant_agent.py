"""Assistant agent — selects tools and generates tool calls.

Uses structured (JSON mode) output to produce reliable, parseable
actions: tool calls, clarification questions, or final responses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..registry.endpoint_semantics import resolve_endpoint_fqn
from ..registry.models import Tool, Endpoint
from ..registry.registry import ToolRegistry
from ..execution.session import SessionState
from ..utils.llm import LLMClient
from .prompts import ASSISTANT_AGENT_SYSTEM, ASSISTANT_AGENT_PROMPT

logger = logging.getLogger(__name__)


class AssistantAction:
    """Parsed action from the assistant agent."""

    def __init__(
        self,
        action_type: str,
        message: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning: str = "",
    ) -> None:
        self.action_type = action_type  # "call_tool" | "ask_clarification" | "respond"
        self.message = message
        self.tool_calls = tool_calls or []
        self.reasoning = reasoning

    @property
    def is_tool_call(self) -> bool:
        return self.action_type == "call_tool" and len(self.tool_calls) > 0

    @property
    def is_clarification(self) -> bool:
        return self.action_type == "ask_clarification"

    @property
    def is_response(self) -> bool:
        return self.action_type == "respond"


class AssistantAgent:
    """AI assistant that decides when and how to call tools, using structured output."""

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ToolRegistry,
    ) -> None:
        self.llm = llm_client
        self.registry = registry
        self._available_tools: list[str] = []

    def set_available_tools(self, endpoint_fqns: list[str]) -> None:
        """Set which tools/endpoints are available for this conversation."""
        self._available_tools = endpoint_fqns

    async def decide(
        self,
        history: list[dict[str, Any]],
        session: SessionState,
        instruction: str = "",
    ) -> AssistantAction:
        """Decide the next action using structured JSON output."""
        tool_desc = self._format_tool_descriptions()
        history_text = self._format_history(history)
        session_ctx = session.get_context_summary()

        if not instruction:
            instruction = "Decide your next action based on the conversation and available tools."

        prompt = ASSISTANT_AGENT_PROMPT.format(
            tool_descriptions=tool_desc,
            history=history_text,
            session_context=session_ctx,
            instruction=instruction,
        )

        result = await self.llm.complete_structured(
            prompt,
            system_prompt=ASSISTANT_AGENT_SYSTEM,
            temperature=0.3,
        )

        return self._parse_action(result)

    def _format_tool_descriptions(self) -> str:
        lines = []
        for fqn in self._available_tools:
            result = self.registry.get_endpoint(fqn)
            if result is None:
                continue
            tool, ep = result
            params_desc = []
            for p in ep.parameters:
                req = "required" if p.required else "optional"
                params_desc.append(f"    - {p.name} ({p.param_type.value}, {req}): {p.description}")
            params_str = "\n".join(params_desc) if params_desc else "    (no parameters)"
            lines.append(
                f"  {fqn}: {ep.description}\n"
                f"    Method: {ep.method}\n"
                f"    Parameters:\n{params_str}"
            )
        return "\n".join(lines) if lines else "(no tools available)"

    @staticmethod
    def _format_history(history: list[dict[str, Any]]) -> str:
        lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content")
            if msg.get("tool_calls"):
                calls = msg["tool_calls"]
                call_strs = []
                for tc in calls:
                    call_strs.append(f"{tc.get('endpoint', '?')}({json.dumps(tc.get('arguments', {}))})")
                lines.append(f"assistant: [Tool calls: {'; '.join(call_strs)}]")
            elif role == "tool":
                content_str = json.dumps(content, default=str) if isinstance(content, dict) else str(content)
                if len(content_str) > 300:
                    content_str = content_str[:300] + "..."
                lines.append(f"tool: {content_str}")
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_action(self, result: dict[str, Any]) -> AssistantAction:
        """Parse the structured JSON response into an AssistantAction."""
        action_type = result.get("action", "respond")
        message = result.get("message")
        tool_calls = result.get("tool_calls", [])
        reasoning = result.get("reasoning", "")

        # Validate tool calls — only accept endpoints in the sampled chain
        available = set(self._available_tools)
        valid_calls = []
        for tc in tool_calls:
            endpoint = tc.get("endpoint") or tc.get("tool", "")
            arguments = tc.get("arguments")
            if arguments is None:
                arguments = tc.get("args", {})
            if not isinstance(arguments, dict):
                arguments = {}
            if endpoint and isinstance(arguments, dict):
                ep = resolve_endpoint_fqn(str(endpoint).strip(), self.registry)
                if available and ep not in available:
                    logger.debug("Rejected tool call %s — not in sampled chain", ep)
                    continue
                valid_calls.append({"endpoint": ep, "arguments": arguments})

        if action_type == "call_tool" and not valid_calls:
            action_type = "respond"
            if not message:
                message = "I'll help you with that."

        # One turn = either user-facing text OR tool calls (not both); avoids noisy training data.
        if action_type == "call_tool" and valid_calls:
            message = None

        return AssistantAction(
            action_type=action_type,
            message=message,
            tool_calls=valid_calls,
            reasoning=reasoning,
        )
