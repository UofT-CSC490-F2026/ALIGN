"""Normalize CooperBench feature pairs or ALIGN-authored scenarios to tasks.jsonl.

Offline only. Does not read from S3 or write to PostgreSQL.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
CONFLICT_TYPES = {"none", "execution_conflict", "design_conflict", "adversarial"}
EXPECTED_OUTCOMES = {"complete_all_goals", "pick_one_goal", "escalate"}
REPOSITORIES = {
    "dottxt_ai_outlines_task": "dottxt-ai/outlines",
    "dspy_task": "stanfordnlp/dspy",
    "go_chi_task": "go-chi/chi",
    "huggingface_datasets_task": "huggingface/datasets",
    "llama_index_task": "run-llama/llama_index",
    "openai_tiktoken_task": "openai/tiktoken",
    "pallets_click_task": "pallets/click",
    "pallets_jinja_task": "pallets/jinja",
    "pillow_task": "python-pillow/Pillow",
    "react_hook_form_task": "react-hook-form/react-hook-form",
    "samuelcolvin_dirty_equals_task": "samuelcolvin/dirty-equals",
    "typst_task": "typst/typst",
}


def cooperbench_task_key(repo: str, task_id: int, a: int, b: int,
                         revision: str) -> str:
    """Identity shared by normalized tasks, trajectory splits, and evaluation."""
    first, second = sorted((a, b))
    return f"cooperbench:{repo}:task{task_id}:f{first}-f{second}:{revision}"


def canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_task(task: dict) -> None:
    """Check the database-facing v1 task and embedded task_goals contract."""
    require(task.get("schema_version") == SCHEMA_VERSION, "Unsupported task schema_version")
    for key in ("task_key", "dataset_source", "external_task_id", "repo_url", "base_commit", "shared_goal"):
        require(isinstance(task.get(key), str) and bool(task[key].strip()), f"Missing {key}")
    require(re.fullmatch(r"[0-9a-f]{40}", task["base_commit"]) is not None, "base_commit must be a full Git SHA")
    require(task["repo_url"].startswith("https://github.com/"), "repo_url must be a GitHub HTTPS URL")
    require(type(task.get("version")) is int and task["version"] > 0, "version must be a positive integer")
    require(task.get("conflict_type") in CONFLICT_TYPES, "Invalid conflict_type")
    require(task.get("expected_outcome") in EXPECTED_OUTCOMES, "Invalid expected_outcome")
    for key in ("constraints", "setup_spec", "metadata"):
        require(isinstance(task.get(key), dict), f"{key} must be an object")
    goals = task.get("goals")
    require(isinstance(goals, list) and bool(goals), "At least one goal is required")
    seen = set()
    for goal in goals:
        require(isinstance(goal, dict), "Goal must be an object")
        goal_key = goal.get("goal_key")
        require(isinstance(goal_key, str) and bool(goal_key.strip()), "Missing goal_key")
        require(goal_key not in seen, f"Duplicate goal_key: {goal_key}")
        seen.add(goal_key)
        owner = goal.get("owner_type")
        actor = goal.get("owner_agent_id")
        require(owner in {"agent", "shared"}, f"Invalid owner_type for {goal_key}")
        require((owner == "agent" and isinstance(actor, str) and bool(actor.strip())) or
                (owner == "shared" and actor is None), f"Invalid owner_agent_id for {goal_key}")
        require(isinstance(goal.get("description"), str) and bool(goal["description"].strip()),
                f"Missing goal description for {goal_key}")
        require(isinstance(goal.get("success_criteria"), dict), f"Missing success_criteria for {goal_key}")
        require(type(goal.get("is_critical")) is bool, f"is_critical must be a boolean for {goal_key}")


def load_subset(dataset_dir: Path, subset: str) -> list[tuple[str, int, int, int]]:
    """Return task-feature pairs. 'all' enumerates all feature combinations in the file tree."""
    pairs = []
    if subset == "all":
        for repo_dir in sorted(dataset_dir.iterdir()):
            if not repo_dir.is_dir() or not repo_dir.name.endswith("_task"):
                continue
            require(repo_dir.name in REPOSITORIES, f"Unknown CooperBench repository: {repo_dir.name}")
            for task_dir in sorted(repo_dir.iterdir()):
                if not task_dir.is_dir() or not re.fullmatch(r"task\d+", task_dir.name):
                    continue
                features = sorted(int(p.name[7:]) for p in task_dir.iterdir()
                                  if p.is_dir() and re.fullmatch(r"feature\d+", p.name))
                pairs.extend((repo_dir.name, int(task_dir.name[4:]), a, b)
                             for a, b in itertools.combinations(features, 2))
    else:
        subset_file = dataset_dir / "subsets" / f"{subset}.json"
        require(subset_file.is_file(), f"Missing CooperBench subset: {subset_file}")
        info = json.loads(subset_file.read_text(encoding="utf-8"))
        require(info.get("name") == subset, "Subset name does not match filename")
        for item in info["tasks"]:
            repo, task_id = item["repo"], item["task_id"]
            require(repo in REPOSITORIES, f"Unknown CooperBench repository: {repo}")
            require(type(task_id) is int and task_id >= 0, "Invalid CooperBench task ID")
            for pair in item["pairs"]:
                require(isinstance(pair, list) and len(pair) == 2 and
                        all(type(i) is int and i > 0 for i in pair) and pair[0] != pair[1],
                        f"Invalid CooperBench feature pair: {pair}")
                a, b = sorted(pair)
                pairs.append((repo, task_id, a, b))
    require(len(pairs) == len(set(pairs)), "Duplicate feature pairs in subset")
    require(bool(pairs), "No CooperBench feature pairs found")
    return sorted(pairs)


def verify_raw_manifest(dataset_dir: Path, revision: str, manifest_path: Path) -> None:
    """Optionally verify a local PR-1 snapshot before deriving task metadata."""
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(original.get("source") == "cooperbench" and
            original.get("source_repository") == "CodeConflict/cooperbench-dataset" and
            original.get("source_revision") == revision, "Raw snapshot provenance does not match revision")
    entries = original.get("files")
    require(isinstance(entries, list) and original.get("file_count") == len(entries),
            "Invalid raw snapshot file inventory")
    actual_files = {f"dataset/{p.relative_to(dataset_dir).as_posix()}" for p in dataset_dir.rglob("*")
                    if p.is_file() and not p.is_symlink()
                    and not any(part in {".cache", ".git", "__pycache__"} for part in p.relative_to(dataset_dir).parts)}
    declared_files = {entry["path"] for entry in entries}
    require(actual_files == declared_files and len(declared_files) == len(entries),
            "Raw snapshot file list does not match dataset directory")
    for entry in entries:
        require(isinstance(entry.get("sha256"), str) and isinstance(entry.get("size_bytes"), int),
                "Malformed raw manifest entry")
        path = dataset_dir / entry["path"].removeprefix("dataset/")
        require(path.is_file() and not path.is_symlink() and
                path.stat().st_size == entry["size_bytes"] and
                sha256_bytes(path.read_bytes()) == entry["sha256"],
                f"Raw snapshot checksum mismatch: {entry['path']}")


def normalized_cooperbench(dataset_dir: Path, revision: str, subset: str) -> list[dict]:
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
            "CooperBench revision must be a pinned 40-character dataset SHA")
    records = []
    for repo, task_id, a, b in load_subset(dataset_dir, subset):
        task_dir = dataset_dir / repo / f"task{task_id}"
        require(task_dir.is_dir(), f"Missing task directory: {task_dir}")
        required = ("Dockerfile", "setup.sh", "runner.sh", "run_tests.sh", "combined.patch")
        for name in required:
            require((task_dir / name).is_file(), f"Missing task file: {task_dir / name}")
        setup = (task_dir / "setup.sh").read_text(encoding="utf-8")
        commits = re.findall(r'^\s*BASE_COMMIT=["\']?([0-9a-f]{40})["\']?\s*$', setup, re.M)
        require(len(commits) == 1, f"Expected exactly one BASE_COMMIT in {task_dir / 'setup.sh'}")
        rel = f"dataset/{repo}/task{task_id}"
        goals = []
        evaluation_only = [f"{rel}/combined.patch"]
        for agent_index, feature_id in enumerate((a, b), start=1):
            feature_path = task_dir / f"feature{feature_id}"
            for filename in ("feature.md", "feature.patch", "tests.patch"):
                require((feature_path / filename).is_file(), f"Missing feature file: {feature_path / filename}")
            feature_rel = f"{rel}/feature{feature_id}"
            evaluation_only.extend((f"{feature_rel}/feature.patch", f"{feature_rel}/tests.patch"))
            goals.append({
                "goal_key": f"feature{feature_id}",
                "owner_type": "agent",
                "owner_agent_id": f"agent{agent_index}",
                "description": (feature_path / "feature.md").read_text(encoding="utf-8").strip(),
                "success_criteria": {"evaluator_test_patch": f"{feature_rel}/tests.patch"},
                "is_critical": True,
            })
        pair_id = f"{repo}:task{task_id}:f{a}-f{b}"
        record = {
            "schema_version": SCHEMA_VERSION,
            "task_key": cooperbench_task_key(repo, task_id, a, b, revision),
            "dataset_source": "cooperbench",
            "external_task_id": f"{pair_id}@{revision}",
            "version": 1,
            "repo_url": f"https://github.com/{REPOSITORIES[repo]}",
            "base_commit": commits[0],
            "shared_goal": "Implement and integrate both features; satisfy both feature test suites.",
            "conflict_type": "none",  # No ALIGN-injected conflict label; emergent conflicts are still possible.
            "expected_outcome": "complete_all_goals",
            "constraints": {},
            "setup_spec": {
                "task_root": rel,
                "dockerfile": f"{rel}/Dockerfile",
                "setup_script": f"{rel}/setup.sh",
                "runner_script": f"{rel}/runner.sh",
                "test_script": f"{rel}/run_tests.sh",
            },
            "metadata": {
                "source_revision": revision,
                "cooperbench_task_id": task_id,
                "cooperbench_repo_directory": repo,
                "feature_ids": [a, b],
                "scenario_type": "original",
                "evaluation_only_artifacts": evaluation_only,
            },
            "goals": goals,
        }
        validate_task(record)
        records.append(record)
    return records


def normalized_custom(scenarios_dir: Path, revision: str) -> list[dict]:
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", revision) is not None,
            "Invalid custom revision")
    require(scenarios_dir.is_dir(), f"Missing scenario directory: {scenarios_dir}")
    files = sorted(scenarios_dir.glob("*.json"))
    require(bool(files), f"No custom JSON scenarios in: {scenarios_dir}")
    records = []
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = item.get("scenario_id")
        require(isinstance(scenario_id, str) and
                re.fullmatch(r"[a-z0-9][a-z0-9._-]*", scenario_id) is not None,
                f"Invalid scenario_id in {path.name}")
        version = item.get("version")
        require(type(version) is int and version > 0, f"Invalid version in {path.name}")
        require(isinstance(item.get("metadata", {}), dict), f"metadata must be an object in {path.name}")
        record = {
            "schema_version": SCHEMA_VERSION,
            "task_key": f"align-custom:{scenario_id}:v{version}:{revision}",
            "dataset_source": "align-custom",
            "external_task_id": f"{scenario_id}@{revision}",
            "version": version,
            "repo_url": item.get("repo_url"),
            "base_commit": item.get("base_commit"),
            "shared_goal": item.get("shared_goal"),
            "conflict_type": item.get("conflict_type"),
            "expected_outcome": item.get("expected_outcome"),
            "constraints": item.get("constraints", {}),
            "setup_spec": item.get("setup_spec", {}),
            "metadata": {**item.get("metadata", {}), "source_revision": revision,
                         "scenario_id": scenario_id, "scenario_type": "custom"},
            "goals": item.get("goals"),
        }
        validate_task(record)
        records.append(record)
    require(len({r["task_key"] for r in records}) == len(records), "Duplicate custom scenario_id/version")
    return records


def write_output(records: list[dict], outdir: Path, source: str, revision: str, selection: str) -> bool:
    """Write deterministic output. Never overwrite a completed differing snapshot."""
    records = sorted(records, key=lambda r: r["task_key"])
    keys = [record["task_key"] for record in records]
    require(len(keys) == len(set(keys)), "Duplicate task_key")
    tasks_bytes = ("\n".join(canonical_json(record) for record in records) + "\n").encode("utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_source": source,
        "source_revision": revision,
        "selection": selection,
        "task_count": len(records),
        "tasks_sha256": sha256_bytes(tasks_bytes),
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    outdir.mkdir(parents=True, exist_ok=True)
    tasks_file, manifest_file = outdir / "tasks.jsonl", outdir / "manifest.json"
    if manifest_file.exists():
        require(manifest_file.read_bytes() == manifest_bytes, "Existing manifest differs; use a new output directory")
        require(tasks_file.is_file() and tasks_file.read_bytes() == tasks_bytes,
                "Completed snapshot has missing or modified tasks.jsonl")
        return False
    if tasks_file.exists():
        require(tasks_file.read_bytes() == tasks_bytes,
                "Incomplete snapshot has conflicting tasks.jsonl")
    else:
        _atomic_write(tasks_file, tasks_bytes)
    _atomic_write(manifest_file, manifest_bytes)  # completion marker last
    return True


def _atomic_write(dest: Path, contents: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".align-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)
    cb = sub.add_parser("cooperbench", help="Normalize official CooperBench feature pairs")
    cb.add_argument("--dataset-dir", required=True, type=Path, help="Local CooperBench dataset/ root")
    cb.add_argument("--revision", required=True, help="Pinned source dataset commit SHA")
    cb.add_argument("--subset", choices=("flash", "lite", "all"), default="flash")
    cb.add_argument("--raw-manifest", type=Path, help="Optional PR-1 raw manifest.json; verify all checksums")
    cb.add_argument("--output-dir", required=True, type=Path)
    custom = sub.add_parser("custom", help="Normalize hand-authored ALIGN JSON scenarios")
    custom.add_argument("--scenarios-dir", required=True, type=Path)
    custom.add_argument("--revision", required=True, help="Explicit scenario-set revision, e.g. v1")
    custom.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.source == "cooperbench":
        if args.raw_manifest:
            verify_raw_manifest(args.dataset_dir, args.revision, args.raw_manifest)
        records = normalized_cooperbench(args.dataset_dir, args.revision, args.subset)
        selection = args.subset
    else:
        records = normalized_custom(args.scenarios_dir, args.revision)
        selection = "all"
    written = write_output(records, args.output_dir, args.source if args.source != "custom" else "align-custom",
                           args.revision, selection)
    print(f"{'Wrote' if written else 'Already present:'} {len(records)} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
