"""Conversation plan intermediate representation.

Sits between the scenario planner and the turn-by-turn orchestrator,
giving a typed, inspectable structure that:
- Maps each expected step to a concrete endpoint
- Records data-flow dependencies between steps
- Tracks execution progress during generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannedStep:
    """A single planned tool-call step."""

    index: int
    endpoint_fqn: str
    description: str = ""
    depends_on: list[int] = field(default_factory=list)
    expected_output_keys: list[str] = field(default_factory=list)

    # Filled during execution
    executed: bool = False
    actual_arguments: dict[str, Any] = field(default_factory=dict)
    actual_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationPlan:
    """Full plan for a conversation, built from scenario + tool chain."""

    scenario: str
    user_persona: str
    initial_request: str
    missing_info: list[str]
    reference_date: str
    steps: list[PlannedStep] = field(default_factory=list)

    @classmethod
    def from_scenario(
        cls,
        scenario_dict: dict[str, Any],
        endpoint_fqns: list[str],
    ) -> ConversationPlan:
        """Build a plan from the LLM scenario planner output and the sampled chain."""
        expected_steps = scenario_dict.get("expected_steps", [])
        steps: list[PlannedStep] = []

        for i, fqn in enumerate(endpoint_fqns):
            desc = expected_steps[i] if i < len(expected_steps) else ""
            depends_on = list(range(i)) if i > 0 else []
            steps.append(
                PlannedStep(
                    index=i,
                    endpoint_fqn=fqn,
                    description=desc,
                    depends_on=depends_on[:1],  # direct dependency on previous step
                )
            )

        return cls(
            scenario=scenario_dict.get("scenario", ""),
            user_persona=scenario_dict.get("user_persona", "a general user"),
            initial_request=scenario_dict.get("initial_request", "I need help"),
            missing_info=scenario_dict.get("missing_info", []),
            reference_date=scenario_dict.get("simulation_reference_date", ""),
            steps=steps,
        )

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.executed)

    @property
    def next_step(self) -> PlannedStep | None:
        for s in self.steps:
            if not s.executed:
                return s
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.executed for s in self.steps)

    def mark_executed(
        self,
        endpoint_fqn: str,
        arguments: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        """Record that a step has been executed."""
        for s in self.steps:
            if s.endpoint_fqn == endpoint_fqn and not s.executed:
                s.executed = True
                s.actual_arguments = arguments
                s.actual_response = response
                break

    def progress_summary(self) -> str:
        """Human-readable progress for assistant instructions."""
        lines = []
        for s in self.steps:
            status = "done" if s.executed else "pending"
            lines.append(f"  Step {s.index + 1} [{status}]: {s.endpoint_fqn}")
            if s.description:
                lines.append(f"    → {s.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for metadata / debugging."""
        return {
            "scenario": self.scenario,
            "user_persona": self.user_persona,
            "initial_request": self.initial_request,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "steps": [
                {
                    "index": s.index,
                    "endpoint": s.endpoint_fqn,
                    "description": s.description,
                    "executed": s.executed,
                }
                for s in self.steps
            ],
        }
