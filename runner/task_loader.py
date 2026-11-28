"""A2 task ingestion and explicit decision-time projection."""
import json
from pathlib import Path

from scripts.normalize_tasks import validate_task

PUBLIC_CONSTRAINTS = frozenset({
    "allowed_paths_by_agent", "protected_paths", "hard_constraints", "forbidden_actions",
})


def agent_view(task):
    goals = {}
    for goal in task["goals"]:
        if goal["owner_type"] == "agent":
            goals.setdefault(goal["owner_agent_id"], []).append(goal["description"])
    if set(goals) != {"agent1", "agent2"}:
        raise ValueError("Exactly agent1 and agent2 are required")
    return {
        "task_key": task["task_key"],
        "shared_goal": task["shared_goal"],
        "agent_goals": {actor: "\n".join(items) for actor, items in goals.items()},
        "constraints": {
            key: value for key, value in task["constraints"].items()
            if key in PUBLIC_CONSTRAINTS
        },
    }


def load_tasks(path):
    tasks = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not tasks or len({task["task_key"] for task in tasks}) != len(tasks):
        raise ValueError("Empty or duplicate tasks")
    for task in tasks:
        validate_task(task)
        agent_view(task)
    return tasks


def cooperbench_identity(task):
    if task["dataset_source"] != "cooperbench":
        raise ValueError("Not a CooperBench task")
    metadata = task["metadata"]
    repo = metadata["cooperbench_repo_directory"]
    number = metadata["cooperbench_task_id"]
    features = metadata["feature_ids"]
    if (
        not isinstance(repo, str) or not repo or "/" in repo or "\\" in repo
        or repo in {".", ".."}
        or type(number) is not int or number < 0
        or not isinstance(features, list) or len(features) != 2
        or any(type(feature) is not int or feature < 1 for feature in features)
        or features[0] >= features[1]
    ):
        raise ValueError("Invalid CooperBench identity")
    if not all(
        any(goal["goal_key"] == f"feature{feature}"
            and goal["owner_agent_id"] == f"agent{index + 1}"
            for goal in task["goals"])
        for index, feature in enumerate(features)
    ):
        raise ValueError("Feature assignment mismatch")
    return repo, number, features
