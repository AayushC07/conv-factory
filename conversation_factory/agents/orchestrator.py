"""Multi-agent conversation orchestrator.

Coordinates the user agent, assistant agent, and mock executor to produce
complete tool-use conversations. Handles:
- Scenario planning (via LLM with structured output)
- Turn-by-turn conversation generation
- Tool execution and result injection
- Conversation grounding (referencing prior outputs)
- Termination conditions
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from ..context.grounding import ConversationGrounding
from ..execution.tool_call_validation import validate_before_execute
from ..execution.mock_executor import MockExecutor
from ..execution.session import SessionState
from ..graph.sampler import ToolChain
from ..registry.registry import ToolRegistry
from ..utils.llm import LLMClient
from .assistant_agent import AssistantAction, AssistantAgent
from .conversation_plan import ConversationPlan
from .orchestrator_turn_policy import (
    action_includes_booking,
    clarify_before_booking_action,
    user_message_after_last_listing_search,
)
from .prompts import SCENARIO_PLANNER_SYSTEM, SCENARIO_PLANNER_PROMPT
from .user_agent import UserAgent

logger = logging.getLogger(__name__)


def _tool_call_signature(endpoint: str, arguments: dict[str, Any]) -> str:
    return f"{endpoint}|{json.dumps(arguments, sort_keys=True, default=str)}"


class ConversationOrchestrator:
    """Orchestrates multi-agent conversation generation."""

    def __init__(
        self,
        registry: ToolRegistry,
        executor: MockExecutor,
        llm_client: LLMClient,
        max_turns: int = 15,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.llm = llm_client
        self.max_turns = max_turns
        self.user_agent = UserAgent(llm_client)
        self.assistant_agent = AssistantAgent(llm_client, registry)
        self.grounding = ConversationGrounding()

    async def generate_conversation(
        self,
        tool_chain: ToolChain,
        conversation_id: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate a complete conversation from a tool chain."""
        conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        session = SessionState()

        # Plan the scenario
        scenario = await self._plan_scenario(tool_chain, seed=seed)
        plan = ConversationPlan.from_scenario(scenario, tool_chain.endpoints)

        # Configure agents
        self.user_agent.configure(
            persona=plan.user_persona,
            scenario=plan.scenario,
            initial_request=plan.initial_request,
            missing_info=plan.missing_info,
        )
        self.assistant_agent.set_available_tools(tool_chain.endpoints)

        # Generate turn by turn
        messages: list[dict[str, Any]] = []
        tools_used: list[str] = []
        num_tool_calls = 0
        num_clarifications = 0
        aborted_redundant_identical_tools = False

        # Consecutive identical (endpoint + grounded args) — block 3rd execution (runaway loop)
        streak_last: str | None = None
        streak_count = 0

        def _should_abort_third_identical(sig: str) -> bool:
            return sig == streak_last and streak_count >= 2

        def _streak_after_execute(sig: str) -> None:
            nonlocal streak_last, streak_count
            if sig == streak_last:
                streak_count += 1
            else:
                streak_last = sig
                streak_count = 1

        # User's opening message
        user_msg = self.user_agent.get_initial_message()
        messages.append({"role": "user", "content": user_msg})

        for turn in range(self.max_turns):
            # If all planned tools have been called, force the assistant to
            # produce a final summary instead of continuing with more tool calls.
            chain_exhausted = num_tool_calls >= len(tool_chain.endpoints)

            # Assistant decides
            action = await self.assistant_agent.decide(
                history=messages,
                session=session,
                instruction=self.grounding.get_assistant_instruction(
                    session, tool_chain, turn, num_tool_calls=num_tool_calls
                ),
            )

            # Override: if chain is exhausted, convert any further tool calls
            # into a final response so we don't inflate tool counts.
            if chain_exhausted and action.is_tool_call:
                action = AssistantAction(
                    action_type="respond",
                    message=action.message or "I've completed all the steps. Let me know if you need anything else.",
                    reasoning="All planned tools have been called.",
                )

            if (
                action.is_tool_call
                and action_includes_booking(action)
                and not user_message_after_last_listing_search(messages)
            ):
                logger.info(
                    "Requiring user choice: booking blocked after listing search without user follow-up"
                )
                action = clarify_before_booking_action()

            if action.is_tool_call:
                prepared: list[tuple[str, dict[str, Any]]] = []
                for tc in action.tool_calls:
                    endpoint = tc["endpoint"]
                    arguments = self.grounding.ground_arguments(
                        endpoint, tc["arguments"], session
                    )
                    prepared.append((endpoint, arguments))

                blocked = False
                for endpoint, arguments in prepared:
                    sig = _tool_call_signature(endpoint, arguments)
                    if _should_abort_third_identical(sig):
                        logger.warning(
                            "Stopping runaway: same tool+args would execute a 3rd consecutive time: %s",
                            endpoint,
                        )
                        aborted_redundant_identical_tools = True
                        messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "I already retrieved this information in the previous step; "
                                    "please refer to the tool results above rather than fetching again."
                                ),
                            }
                        )
                        blocked = True
                        break

                if blocked:
                    break

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": action.tool_calls,
                })

                for endpoint, arguments in prepared:
                    sig = _tool_call_signature(endpoint, arguments)
                    validation_error = validate_before_execute(
                        endpoint, arguments, session, registry=self.registry
                    )
                    if validation_error is not None:
                        response = validation_error
                        session.record_call(endpoint, arguments, response)
                    else:
                        response = await self.executor.execute(endpoint, arguments, session)
                    messages.append({"role": "tool", "content": response})
                    tools_used.append(endpoint)
                    num_tool_calls += 1
                    _streak_after_execute(sig)
                    plan.mark_executed(endpoint, arguments, response)

                continue

            elif action.is_clarification:
                messages.append({"role": "assistant", "content": action.message})
                num_clarifications += 1

                # User responds to clarification
                user_response = await self.user_agent.respond(
                    history=messages,
                    instruction="The assistant asked you a clarifying question. Provide the requested information.",
                )
                messages.append({"role": "user", "content": user_response})

            elif action.is_response:
                messages.append({"role": "assistant", "content": action.message})

                # Check if the conversation should end
                if self._should_end(messages, num_tool_calls, tool_chain):
                    break

                # User responds or asks follow-up
                if turn < self.max_turns - 2:
                    # If there are remaining tools, steer the user to naturally
                    # prompt the next tool call rather than ending the conversation.
                    remaining_tools = tool_chain.endpoints[num_tool_calls:]
                    if remaining_tools and num_tool_calls > 0:
                        next_ep = remaining_tools[0]
                        ep_info = self.registry.get_endpoint(next_ep)
                        ep_desc = ep_info[1].description if ep_info else next_ep
                        user_instruction = (
                            f"The assistant has helped with part of your task, but you still need: "
                            f"{ep_desc}. Ask a natural follow-up that would require this next step. "
                            f"Do NOT mention API names."
                        )
                    elif remaining_tools and num_tool_calls == 0:
                        user_instruction = (
                            "The assistant hasn't taken action yet. Provide more details about "
                            "what you need so the assistant can use the right tools to help you."
                        )
                    else:
                        user_instruction = (
                            "React to the assistant's response. You can thank them, "
                            "ask a follow-up, or end the conversation."
                        )
                    user_response = await self.user_agent.respond(
                        history=messages,
                        instruction=user_instruction,
                    )
                    messages.append({"role": "user", "content": user_response})
                else:
                    break

        meta: dict[str, Any] = {
            "seed": seed,
            "tools_used": list(set(tools_used)),
            "num_turns": len(messages),
            "num_tool_calls": num_tool_calls,
            "num_clarifications": num_clarifications,
            "num_distinct_tools": len(set(tools_used)),
            "categories": tool_chain.metadata.get("categories", []),
            "tool_chain_pattern": tool_chain.pattern,
            "scenario": plan.scenario,
            "plan": plan.to_dict(),
        }
        if tool_chain.pattern == "parallel":
            meta["parallel_chain_note"] = (
                "Sampler marked this chain as parallel. The assistant may emit multiple "
                "tool_calls in one turn; the executor still runs calls in order and emits "
                "one tool message per call."
            )
        meta["simulation_reference_date"] = plan.reference_date
        if aborted_redundant_identical_tools:
            meta["aborted_redundant_identical_tools"] = True

        return {
            "conversation_id": conv_id,
            "messages": messages,
            "metadata": meta,
        }

    def _reference_date_for_seed(self, seed: int | None) -> str:
        """Deterministic calendar anchor so scenarios do not mix unrelated years."""
        from datetime import date, timedelta

        base = date(2026, 1, 1)
        if seed is None:
            return base.isoformat()
        offset = int(seed) % 500
        return (base + timedelta(days=offset)).isoformat()

    async def _plan_scenario(
        self, tool_chain: ToolChain, seed: int | None = None
    ) -> dict[str, Any]:
        """Use LLM (structured output) to plan a realistic scenario for the tool chain."""
        tool_descriptions = []
        for fqn in tool_chain.endpoints:
            result = self.registry.get_endpoint(fqn)
            if result is None:
                tool_descriptions.append(f"- {fqn}: (unknown endpoint)")
                continue
            tool, ep = result
            params = ", ".join(
                f"{p.name}({'required' if p.required else 'optional'})"
                for p in ep.parameters
            )
            tool_descriptions.append(
                f"- {fqn}: {ep.description} | Params: {params}"
            )

        needs_disambiguation = len(tool_chain.endpoints) >= 3
        disambiguation_instruction = (
            "Make the initial request somewhat vague so the assistant needs to ask a clarifying question"
            if needs_disambiguation
            else "The user can be fairly specific in their initial request"
        )

        categories = ", ".join(tool_chain.metadata.get("categories", ["general"]))

        reference_date = self._reference_date_for_seed(seed)
        prompt = SCENARIO_PLANNER_PROMPT.format(
            tool_descriptions="\n".join(tool_descriptions),
            num_steps=tool_chain.num_steps,
            disambiguation_instruction=disambiguation_instruction,
            categories=categories,
            reference_date=reference_date,
        )

        result = await self.llm.complete_structured(
            prompt,
            system_prompt=SCENARIO_PLANNER_SYSTEM,
            temperature=0.7,
        )

        # Ensure required fields
        if "initial_request" not in result:
            result["initial_request"] = "I need help with something."
        if "scenario" not in result:
            result["scenario"] = "Complete a multi-step task."
        if "user_persona" not in result:
            result["user_persona"] = "a general user"
        result["simulation_reference_date"] = reference_date

        return result

    def _should_end(
        self,
        messages: list[dict[str, Any]],
        num_tool_calls: int,
        tool_chain: ToolChain,
    ) -> bool:
        """Decide if the conversation should end."""
        if len(messages) >= self.max_turns:
            return True
        if num_tool_calls >= len(tool_chain.endpoints):
            # All planned tools have been called — end the conversation.
            return True
        # Don't end early if there are still tools to call — give the LLM
        # more chances to complete the chain.
        if num_tool_calls < len(tool_chain.endpoints):
            return False
        return False
