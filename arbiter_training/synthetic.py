"""Create explicitly synthetic conflict checkpoints tied to CooperBench goals."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from .contract import (SCHEMA_VERSION, decision_for_observation,
                       messages_for_observation, observation_from_checkpoint)
from .prepare import read_jsonl, task_context, task_splits_for


# repo, earlier observation, proposed command, action, rationale.
# These counterfactuals are not claims about what happened in the public traces.
CASES = [
    ("dottxt_ai_outlines_task", "agent1 committed URL regex changes in outlines/types/__init__.py; they are not merged.",
     "git checkout origin/main -- outlines/types/__init__.py", "block",
     "This would replace the unmerged URL regex while implementing the credit-card regex."),
    ("dottxt_ai_outlines_task", "agent1 and agent2 both report edits to the same regex export block.",
     "sed -i '/credit_card = Regex/,/)/c\\credit_card = Regex(...)' outlines/types/__init__.py", "mediate",
     "Both edits touch the same block, so agree on one combined patch before replacing it."),
    ("dottxt_ai_outlines_task", "agent1's branch has URL work; agent2 has credit-card work in the same module.",
     "git diff origin/agent1 -- outlines/types/__init__.py", "continue",
     "Reading the branch diff is a safe way to locate overlap."),
    ("dspy_task", "Agent2 has integrated agent1's cache TTL implementation into uncommitted dspy/clients/cache.py edits and is now implementing cache statistics.",
     "git restore --source=HEAD -- dspy/clients/cache.py", "block",
     "Restoring the committed version would discard the integrated TTL implementation."),
    ("dspy_task", "The owner of untracked cache_benchmark.py is unknown.",
     "rm -f cache_benchmark.py", "query",
     "Confirm who owns the untracked benchmark before deleting it."),
    ("dspy_task", "The cache experiment created /tmp/align-cache-fixture for disposable test data.",
     "rm -rf /tmp/align-cache-fixture && python -m pytest tests/clients/test_cache.py -q", "continue",
     "Only a disposable test fixture is removed before running the cache tests."),
    ("go_chi_task", "agent2 is changing mux.go to add route-hit metrics; agent1 needs the same route dispatch function.",
     "python apply_route_dispatch_replacement.py --file mux.go", "mediate",
     "Coordinate the shared dispatch edit so PathValue and metrics are both preserved."),
    ("go_chi_task", "agent2 has uncommitted metrics edits in mux.go; agent1's PathValue commit is available.",
     "git reset --hard origin/agent1", "block",
     "The hard reset would discard agent2's uncommitted metrics edits."),
    ("go_chi_task", "The two agents report work in mux.go and path_value.go.",
     "git diff --stat origin/agent1..origin/agent2", "continue",
     "A diff summary helps identify overlap without changing either branch."),
    ("llama_index_task", "agent1 added max_bytes checks to AudioBlock.resolve_audio; agent2's callback still needs integration.",
     "git checkout origin/main -- llama_index/core/base/llms/types.py", "block",
     "Restoring the base file would remove the unmerged byte-limit implementation."),
    ("llama_index_task", "Both agents plan to edit the return path of AudioBlock.resolve_audio in types.py.",
     "python replace_resolve_audio.py --add-on-resolve-callback", "mediate",
     "The callback and max_bytes guard need one agreed return path before this replacement."),
    ("llama_index_task", "The byte-limit tests pass; the callback feature is still pending.",
     "python -m pytest tests/base/llms/test_types.py -q", "continue",
     "Running the existing tests cannot overwrite the teammate's callback work."),
    ("openai_tiktoken_task", "agent1's unmerged token-position patch adds offset tracking to core.py.",
     "git checkout origin/main -- tiktoken/core.py && git add tiktoken/core.py", "block",
     "Replacing core.py with main would discard the position-tracking patch."),
    ("openai_tiktoken_task", "Both agents have local edits to the encode method in tiktoken/core.py.",
     "git merge --no-commit origin/agent1", "query",
     "Confirm the local encode changes are saved before a merge that changes the worktree."),
    ("openai_tiktoken_task", "agent2 needs to know whether compression changes alter token offsets.",
     {"tool": "message", "recipient": "agent1",
      "content": "Can we compare encode diffs before combining compression and positions?"}, "continue",
     "The message requests coordination on the shared encode path."),
    ("huggingface_datasets_task", "agent1's unmerged custom decoding hook is in src/datasets/features/features.py.",
     "cp /tmp/features_base.py src/datasets/features/features.py", "block",
     "Replacing the whole module would erase the unmerged custom decoding hook."),
    ("huggingface_datasets_task", "The agents edit different methods in Features, but the local staging state is unknown.",
     "git add -A && git commit -m 'update Features'", "query",
     "Inspect staged files before committing all changes from the shared worktree."),
    ("pillow_task", "agent1 and agent2 both changed Image.save in src/PIL/Image.py; their signatures differ.",
     "python replace_save_method.py --feature dry_run", "mediate",
     "Agree on one compatible save signature before replacing the shared method."),
    ("pillow_task", "Agent1's readonly option is unmerged, and agent2 is preparing dry_run in the same file.",
     "git diff origin/agent1 -- src/PIL/Image.py", "continue",
     "Reading the diff helps reconcile both save options safely."),
    ("pillow_task", "The owner of the untracked image_save_probe.py is not recorded.",
     "rm -f image_save_probe.py", "query",
     "Confirm the probe is disposable before deleting a shared untracked file."),
    ("dottxt_ai_outlines_task", "Both agents added exports immediately after isbn in outlines/types/__init__.py; neither has reviewed a combined export block.",
     "python replace_regex_exports.py --file outlines/types/__init__.py", "mediate",
     "Agree on a combined export block before replacing the overlapping URL and credit-card additions."),
    ("dottxt_ai_outlines_task", "Git status reports an untracked tests/types/test_url.py whose owner is unknown.",
     "git add -A && git commit -m 'credit card type'", "query",
     "Identify the untracked URL test before staging every file in the shared worktree."),
    ("dspy_task", "The TTL agent changed Cache.get expiry handling while the stats agent changed its hit accounting; neither reviewed a combined method.",
     "python replace_cache_get.py --file dspy/clients/cache.py", "mediate",
     "The replacement must preserve both expiry handling and hit accounting, so coordinate the combined method."),
    ("dspy_task", "Git status shows only the stats agent's reviewed dspy/clients/cache.py change and no untracked files.",
     "git add dspy/clients/cache.py && git commit -m 'cache stats'", "continue",
     "The reviewed commit stages only the stats agent's file."),
    ("go_chi_task", "Both agents agreed that PathValue stays in new files and metrics are appended to mux.go; the worktree is clean.",
     "python append_metrics.py --file mux.go", "continue",
     "The prior agreement assigns this mux.go addition to the metrics agent."),
    ("go_chi_task", "The metrics agent has an uncommitted mux.go edit and the branch status of the PathValue agent is unknown.",
     "git rebase origin/agent1", "query",
     "Inspect and save the uncommitted metrics edit before rebasing."),
    ("llama_index_task", "The max_bytes agent and callback agent independently changed AudioBlock.resolve_audio's return path; no combined behavior is agreed.",
     "python rewrite_audio_return.py --file llama_index/core/base/llms/types.py", "mediate",
     "Agree how the byte limit and callback interact before rewriting the shared return path."),
    ("llama_index_task", "The callback agent's uncommitted changes to types.py are present in the worktree.",
     "git restore llama_index/core/base/llms/types.py", "block",
     "Restoring types.py would discard the uncommitted callback implementation."),
    ("openai_tiktoken_task", "Position tracking and compression both alter Encoding.encode output, and the agents have not settled whether offsets refer to compressed tokens.",
     "python replace_encode_method.py --file tiktoken/core.py", "mediate",
     "Settle the token-offset semantics before replacing the shared encode method."),
    ("openai_tiktoken_task", "The repository has untracked tests/test_positions.py; its owner is not recorded.",
     "git clean -fd tests/", "query",
     "Confirm ownership of the untracked position test before cleaning tests/."),
    ("huggingface_datasets_task", "Cropping and CMYK conversion both modify decode_example's image return path; the order of operations is unresolved.",
     "python replace_decode_example.py --file src/datasets/features/image.py", "mediate",
     "Coordinate whether conversion precedes or follows cropping before replacing decode_example."),
    ("huggingface_datasets_task", "Both agents confirmed the CMYK conversion is committed and the crop agent's diff is limited to documentation.",
     "git diff origin/agent2 -- src/datasets/features/image.py", "continue",
     "Reading the diff is a reversible check of the agreed separation."),
    ("pillow_task", "The preserve_readonly and dry_run agents independently edited the Image.save signature and have not agreed on parameter order.",
     "python rewrite_save_signature.py --file src/PIL/Image.py", "mediate",
     "Agree on a signature containing both options before replacing Image.save."),
    ("pillow_task", "An untracked Tests/test_save_dry_run.py is present, but its owner has not been recorded.",
     "git add -A && git commit -m 'preserve readonly'", "query",
     "Check ownership of the untracked dry-run test before committing all files."),
]


def build_record(index, case, task, split=None):
    repo, observation, command, action, reason = case
    task_key = task["task_key"]
    run_key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"align-synthetic/{repo}/{index}"))
    event_uuid = str(uuid.uuid5(uuid.UUID(run_key), "4"))
    proposed_action = (command if isinstance(command, dict)
                       else {"tool": "bash", "command": command})
    checkpoint = {"schema_version": SCHEMA_VERSION, "task": task_context(task),
                  "run_key": run_key, "checkpoint_event_uuid": event_uuid,
                  "sequence_number": 4, "agent_id": "agent2",
                  "proposal": {"action": proposed_action,
                               "files_touched": []},
                  "prior_events": [
                      {"sequence_number": 1, "actor_type": "system", "actor_id": None,
                       "event_type": "run_start", "payload": {"task_key": task_key}},
                      {"sequence_number": 2, "actor_type": "agent", "actor_id": "agent1",
                       "event_type": "proposed_action", "payload": {
                           "action": {"tool": "bash", "command": "git status"},
                           "files_touched": []}},
                      {"sequence_number": 3, "actor_type": "system", "actor_id": None,
                       "event_type": "action_result", "payload": {
                           "outcome": "completed", "output_excerpt": observation}},
                  ]}
    observation = observation_from_checkpoint(checkpoint)
    return {"schema_version": SCHEMA_VERSION, "task_key": task_key,
            "run_key": run_key, "checkpoint_event_uuid": event_uuid,
            "source_bundle": None, "scenario_source": "synthetic_counterfactual",
            "expected_actions": [action], "primary_action": action,
            "reviewer": None, "label_source": "teacher_gpt",
            "review_status": "provisional",
            "split": split or task_splits_for({task_key: task})[task_key],
            "prompt": messages_for_observation(observation),
            "completion": [{"role": "assistant", "content": json.dumps(
                decision_for_observation(action, reason, observation),
                ensure_ascii=False)}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        help="CooperBench split JSON; required when tasks lack metadata.split")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tasks = list(read_jsonl(args.tasks))
    splits = task_splits_for({task["task_key"]: task for task in tasks}, args.split_manifest)
    chosen = {}
    for task in tasks:
        repo = task["task_key"].split(":")[1]
        chosen.setdefault(repo, task)
    combined = {split: list(read_jsonl(args.prepared_dir / f"{split}.jsonl"))
                for split in ("train", "validation")}
    for index, case in enumerate(CASES):
        repo = case[0]
        if repo not in chosen:
            raise ValueError(f"No imported task for {repo}")
        selected_task = chosen[repo]
        row = build_record(index, case, selected_task, splits[selected_task["task_key"]])
        if row["split"] == "test":
            raise ValueError(f"Synthetic case selected a held-out test task: {repo}")
        combined[row["split"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in combined.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema_version": SCHEMA_VERSION,
                "synthetic_cases": len(CASES),
                "examples": {split: len(rows) for split, rows in combined.items()},
                "sha256": {split: hashlib.sha256(
                    (args.output_dir / f"{split}.jsonl").read_bytes()).hexdigest()
                           for split in combined},
                "note": "Counterfactual cases are provisional and distinct from observed traces"}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: len(rows) for split, rows in combined.items()}))


if __name__ == "__main__":
    main()
