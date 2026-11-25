"""Convert pinned public CooperBench two-agent traces to A2 event bundles."""

import argparse
import hashlib
import json
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

TRACE_REPO = "CooperBench/qwen35-9b-cooper-flash-evals"
TRACE_REVISION = "bf6cc8469a7b853f70e8f043af145b5053952293"
TRACE_PREFIX = "flash50-qwen3-5-9b/coop/"
TASK_REPO = "CooperBench/cooperbench-dataset"
TASK_REVISION = "b612b1a35af722751454813d9e5a7888f065fc9e"


def download_text(repo, revision, filename):
    path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                           filename=filename)
    return Path(path).read_text(encoding="utf-8")


def iso_time(timestamp):
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def redact(text, limit):
    text = str(text)
    text = re.sub(r"(?i)(api[_-]?key|password|token|secret)(\s*[=:]\s*)\S+",
                  r"\1\2[REDACTED]", text)
    return text[:limit]


def trace_actions(trace, actor):
    results = {}
    for message in trace.get("messages", []):
        if message.get("role") == "tool" and message.get("tool_call_id"):
            results[message["tool_call_id"]] = message
    actions = []
    seen = set()
    for index, message in enumerate(trace.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        timestamp = (message.get("extra") or {}).get("timestamp")
        if timestamp is None:
            continue
        for call in message.get("tool_calls") or []:
            call_id = call.get("id")
            if not call_id or call_id in seen:
                continue
            seen.add(call_id)
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {"raw": function.get("arguments", "")}
            if not isinstance(arguments, dict):
                arguments = {"raw": str(arguments)}
            command = redact(arguments.get("command", arguments), 4000)
            result = results.get(call_id)
            result_extra = (result or {}).get("extra") or {}
            result_content = (result or {}).get("content")
            actions.append({"timestamp": float(timestamp), "actor": actor,
                            "index": index, "call_id": call_id,
                            "tool": function.get("name", "unknown"),
                            "command": command,
                            "result_timestamp": result_extra.get("timestamp"),
                            "returncode": result_extra.get("returncode"),
                            "result": redact(result_content, 1000) if result_content else None})
    return actions


def event(run_key, sequence, actor_type, actor_id, event_type, timestamp, payload,
          trigger_sequence=None):
    value = {"schema_version": 1,
             "event_uuid": str(uuid.uuid5(uuid.UUID(run_key), str(sequence))),
             "run_key": run_key, "sequence_number": sequence,
             "actor_type": actor_type, "actor_id": actor_id,
             "event_type": event_type, "occurred_at": iso_time(timestamp),
             "payload": payload}
    if trigger_sequence is not None:
        value["trigger_sequence"] = trigger_sequence
    return value


def make_bundle(row, paths, output_dir):
    traces = {}
    for actor in ("agent1", "agent2"):
        traces[actor] = json.loads(download_text(TRACE_REPO, TRACE_REVISION, paths[actor]))
    specs = {}
    for actor, feature in (("agent1", row["f1"]), ("agent2", row["f2"])):
        filename = f"{row['repo']}/task{row['task_id']}/feature{feature}/feature.md"
        specs[actor] = download_text(TASK_REPO, TASK_REVISION, filename)
    task = {"task_key": row["task_key"], "repo_url": None, "base_commit": None,
            "shared_goal": "Implement both CooperBench feature specifications so both pass their tests.",
            # The public task does not disclose whether a live conflict exists.
            # Gold-patch merge status is an evaluator-only outcome.
            "conflict_type": None,
            "constraints": {},
            "goals": [{"goal_key": f"feature{row['f1']}", "owner_agent_id": "agent1",
                       "description": specs["agent1"][:7000]},
                      {"goal_key": f"feature{row['f2']}", "owner_agent_id": "agent2",
                       "description": specs["agent2"][:7000]}],
            "metadata": {"source": TRACE_REPO, "trace_revision": TRACE_REVISION,
                         "task_revision": TASK_REVISION, "split": row["split"]}}
    actions = []
    for actor, trace in traces.items():
        actions.extend(trace_actions(trace, actor))
    actions.sort(key=lambda a: (a["timestamp"], a["actor"], a["index"]))
    if not actions:
        raise ValueError(f"No tool proposals for {row['task_key']}")
    run_key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{TRACE_REPO}@{TRACE_REVISION}/{row['task_key']}"))
    started = actions[0]["timestamp"] - 0.001
    events = [event(run_key, 1, "system", None, "run_start", started,
                    {"task_key": row["task_key"], "arbiter_condition": "monitor",
                     "agent_config": {"model": "Qwen/Qwen3.5-9B", "source": TRACE_REPO},
                     "arbiter_config": {}, "seed": None})]
    timeline = []
    for action in actions:
        timeline.append((action["timestamp"], 0, action, "proposal"))
        if action["result_timestamp"] is not None:
            timeline.append((max(float(action["result_timestamp"]), action["timestamp"]),
                             1, action, "result"))
    timeline.sort(key=lambda item: (item[0], item[1], item[2]["actor"], item[2]["index"]))
    proposal_sequences = {}
    last_timestamp = started
    for timestamp, _, action, kind in timeline:
        timestamp = max(timestamp, last_timestamp)
        if kind == "proposal":
            sequence = len(events) + 1
            proposal_sequences[action["call_id"]] = sequence
            events.append(event(run_key, sequence, "agent", action["actor"],
                                "proposed_action", timestamp,
                                {"action": {"tool": action["tool"],
                                            "command": action["command"]},
                                 "files_touched": []}))
        else:
            events.append(event(run_key, len(events) + 1, "system", None,
                                "action_result", timestamp,
                                {"outcome": "completed" if action["returncode"] == 0 else "failed",
                                 "returncode": action["returncode"],
                                 "output_excerpt": action["result"]},
                                trigger_sequence=proposal_sequences[action["call_id"]]))
        last_timestamp = timestamp
    bundle_dir = output_dir / row["split"] / row["task_key"].replace(":", "_")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    raw = ("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in events)).encode("utf-8")
    (bundle_dir / "events.jsonl").write_bytes(raw)
    header = events[0]["payload"]
    manifest = {"schema_version": 1, "run_key": run_key, "task_key": row["task_key"],
                "arbiter_condition": "monitor", "agent_config": header["agent_config"],
                "arbiter_config": {}, "seed": None, "complete": False,
                "event_count": len(events), "events_sha256": hashlib.sha256(raw).hexdigest(),
                "events_size_bytes": len(raw)}
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return task, len(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                        default=["train", "validation"])
    parser.add_argument("--max-pairs-per-split", type=int, default=2)
    args = parser.parse_args()
    if args.max_pairs_per_split < 1:
        parser.error("max-pairs-per-split must be positive")
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split_manifest.get("revision") != TASK_REVISION:
        raise ValueError("Task dataset revision differs from pinned importer revision")
    paths = HfApi().list_repo_files(TRACE_REPO, repo_type="dataset", revision=TRACE_REVISION)
    grouped = defaultdict(dict)
    for path in paths:
        if not path.startswith(TRACE_PREFIX) or not path.endswith("_full_traj.json"):
            continue
        parts = path[len(TRACE_PREFIX):].split("/")
        if len(parts) != 4:
            continue
        repo, task_id, pair, filename = parts
        grouped[(repo, int(task_id), pair)][filename.split("_")[0]] = path
    selected = []
    counts = defaultdict(int)
    available = defaultdict(lambda: defaultdict(list))
    for row in split_manifest["pairs"]:
        split = row["split"]
        if split not in args.splits:
            continue
        pair = f"f{row['f1']}_f{row['f2']}"
        candidate = grouped.get((row["repo"], row["task_id"], pair), {})
        if not {"agent1", "agent2"} <= candidate.keys():
            continue
        available[split][row["repo"]].append((row, candidate))
    # Round-robin repositories so a small pilot is not dominated by the first repo.
    for split in args.splits:
        repo_groups = available[split]
        while counts[split] < args.max_pairs_per_split and any(repo_groups.values()):
            for repo in sorted(repo_groups):
                if counts[split] >= args.max_pairs_per_split:
                    break
                if repo_groups[repo]:
                    selected.append(repo_groups[repo].pop(0))
                    counts[split] += 1
    if any(counts[split] == 0 for split in args.splits):
        raise ValueError("No complete two-agent traces for a requested split")
    tasks = []
    stats = []
    for row, candidate in selected:
        task, proposals = make_bundle(row, candidate, args.output_dir)
        tasks.append(task)
        stats.append({"task_key": row["task_key"], "split": row["split"],
                      "proposals": proposals, "trace_paths": candidate})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tasks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    manifest = {"trace_repo": TRACE_REPO, "trace_revision": TRACE_REVISION,
                "task_repo": TASK_REPO, "task_revision": TASK_REVISION,
                "imported": stats}
    (args.output_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(stats), "proposals": sum(s["proposals"] for s in stats),
                      "by_split": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
