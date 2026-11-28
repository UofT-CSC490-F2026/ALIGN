"""JSON-compatible contracts shared with the model and evaluation workstreams."""
from __future__ import annotations

from typing import Literal, Protocol, TypedDict

Action = Literal["continue", "query", "block", "mediate", "restructure", "escalate"]
Condition = Literal["none", "monitor", "rule_based", "llm"]


class ProposedAction(TypedDict):
    action: dict
    files_touched: list[str]


class ArbiterObservation(TypedDict):
    schema_version: Literal[1]
    task_key: str
    shared_goal: str
    agent_goals: dict[str, str]
    constraints: dict
    actor_id: str
    proposal_sequence: int
    proposed_action: ProposedAction
    recent_events: list[dict]
    repository_state: dict


class ArbiterDecision(TypedDict):
    action: Action
    reason: str
    affected_agents: list[str]
    evidence: dict


class Arbiter(Protocol):
    def decide(self, observation: ArbiterObservation) -> ArbiterDecision:
        """Return a decision; the runner, not the model, enforces it."""
        ...
