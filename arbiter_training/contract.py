"""One inference contract shared by the training and execution harnesses."""

import json

ACTIONS = frozenset({"continue", "query", "block", "mediate", "restructure", "escalate"})
ENFORCED_ACTIONS = ACTIONS - {"restructure"}
SCHEMA_VERSION = 1
SYSTEM_PROMPT = (
    "You are the ALIGN arbiter. Decide what to do before a coding agent's proposed "
    "action executes. Use only the supplied task, proposal, and earlier events. "
    "Return exactly one JSON object with keys action, reason, affected_agents, "
    "and evidence. affected_agents is a list of agent IDs; evidence is an "
    "object citing relevant proposal or prior event sequence numbers. action must be "
    "one of: continue, query, block, mediate, escalate. "
    "Choose continue for a safe or reversible step, including read-only inspection "
    "and an ordinary merge probe on a known clean branch. Choose query when a "
    "missing fact such as file ownership or unsaved work must be checked before "
    "acting. Choose block when the proposal would discard or overwrite known "
    "unintegrated work. Choose mediate when both agents' confirmed changes to "
    "the same behavior need an agreed combined edit before proceeding. Choose "
    "escalate when the agents cannot resolve a substantive goal conflict. "
    "Do not choose restructure: the runner reserves it but cannot enforce it. "
    "Do not infer a conflict merely from sharing a file. Do not assume an "
    "earlier conflict is still open after a conflict_resolution event. "
    "Keep reason concise and name the decisive evidence."
)


def messages_for(checkpoint):
    return messages_for_observation(observation_from_checkpoint(checkpoint))


def observation_from_checkpoint(checkpoint):
    """Map an A2 proposal to the runner's observation envelope.

    Public traces have no reliable repository snapshot, so repository_state is
    empty. The live harness must supply its own repository_state directly.
    """
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    task = checkpoint["task"]
    goals = {}
    for goal in task.get("goals", []):
        owner = goal["owner_agent_id"]
        goals[owner] = (goals.get(owner, "") + "\n" + goal["description"]).strip()
    proposal = checkpoint["proposal"]
    observation = {
        "schema_version": SCHEMA_VERSION,
        "task_key": task["task_key"],
        "shared_goal": task["shared_goal"],
        "agent_goals": goals,
        "constraints": task.get("constraints", {}),
        "actor_id": checkpoint["agent_id"],
        "proposal_sequence": checkpoint["sequence_number"],
        "proposed_action": {
            "action": proposal["action"],
            "files_touched": proposal.get("files_touched", []),
        },
        "recent_events": checkpoint["prior_events"],
        "repository_state": checkpoint.get("repository_state", {}),
    }
    return observation


def messages_for_observation(observation):
    required = {"schema_version", "task_key", "shared_goal", "agent_goals",
                "constraints", "actor_id", "proposal_sequence", "proposed_action",
                "recent_events", "repository_state"}
    if set(observation) != required or observation["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Observation does not match Person 1's schema")
    proposal = observation["proposed_action"]
    if (not isinstance(proposal, dict) or set(proposal) != {"action", "files_touched"}
            or not isinstance(proposal["action"], dict)
            or not isinstance(proposal["files_touched"], list)
            or any(not isinstance(path, str) for path in proposal["files_touched"])):
        raise ValueError("ProposedAction must match runner/contracts.py")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                observation, ensure_ascii=False, sort_keys=True)}]


def decision_for_observation(action, reason, observation):
    actor = observation["actor_id"]
    agents = sorted(observation["agent_goals"])
    if action == "continue":
        affected = []
    elif action == "query":
        affected = [actor]
    else:
        affected = agents
    return {
        "action": action,
        "reason": reason,
        "affected_agents": affected,
        "evidence": {
            "proposal_sequence": observation["proposal_sequence"],
            "recent_event_sequences": [event["sequence_number"]
                                       for event in observation["recent_events"][-2:]],
        },
    }


def parse_decision(output, observation=None, enforceable=False):
    text = output.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Arbiter response is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "action", "reason", "affected_agents", "evidence"
    }:
        raise ValueError("Arbiter response must contain the four decision fields")
    if (not isinstance(value["action"], str) or value["action"] not in ACTIONS
            or not isinstance(value["reason"], str) or not value["reason"].strip()):
        raise ValueError("Arbiter response has invalid action or reason")
    if (not isinstance(value["affected_agents"], list)
            or any(not isinstance(agent, str) for agent in value["affected_agents"])
            or not isinstance(value["evidence"], dict)):
        raise ValueError("Arbiter response has invalid affected_agents or evidence")
    if observation is not None and any(agent not in observation["agent_goals"]
                                       for agent in value["affected_agents"]):
        raise ValueError("Arbiter response names an unknown affected agent")
    if enforceable and value["action"] not in ENFORCED_ACTIONS:
        raise ValueError("Arbiter response selects an unsupported enforced action")
    try:
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Arbiter response is not JSON serializable") from exc
    return value
