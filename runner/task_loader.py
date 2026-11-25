"""A2 task ingestion and explicit decision-time projection."""
import json
from pathlib import Path
from scripts.normalize_tasks import validate_task

def agent_view(task):
    goals = {}
    for g in task["goals"]:
        if g["owner_type"] == "agent":
            goals.setdefault(g["owner_agent_id"], []).append(g["description"])
    if set(goals) != {"agent1", "agent2"}:
        raise ValueError("Exactly agent1 and agent2 are required")
    return {"task_key": task["task_key"], "shared_goal": task["shared_goal"],
            "agent_goals": {k: "\n".join(v) for k,v in goals.items()},
            "constraints": {k:v for k,v in task["constraints"].items() if k in
              {"allowed_paths_by_agent","protected_paths","hard_constraints","forbidden_actions"}}}

def load_tasks(path):
    tasks = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not tasks or len({t["task_key"] for t in tasks}) != len(tasks):
        raise ValueError("Empty or duplicate tasks")
    for task in tasks:
        validate_task(task)
        agent_view(task)
    return tasks

def cooperbench_identity(task):
    if task["dataset_source"] != "cooperbench":
        raise ValueError("Not a CooperBench task")
    m = task["metadata"]
    repo, number, fs = m["cooperbench_repo_directory"], m["cooperbench_task_id"], m["feature_ids"]
    if (not isinstance(repo,str) or not repo or "/" in repo or "\\" in repo or repo in {".",".."}
        or type(number) is not int or number < 1 or not isinstance(fs,list) or len(fs)!=2
        or any(type(f) is not int or f<1 for f in fs) or fs[0]>=fs[1]):
        raise ValueError("Invalid CooperBench identity")
    if not all(any(g["goal_key"]==f"feature{f}" and g["owner_agent_id"]==f"agent{i+1}"
                   for g in task["goals"]) for i,f in enumerate(fs)):
        raise ValueError("Feature assignment mismatch")
    return repo, number, fs
