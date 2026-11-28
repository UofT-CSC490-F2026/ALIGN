"""Offline tests for pinned trace import and annotation command line tools."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("huggingface_hub")

from arbiter_training import adjudicate, import_traces, inspect_queue, pilot_labels
from arbiter_training import split_cooperbench


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def invoke(monkeypatch, module, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, args)])
    module.main()


def test_teacher_first_pass_success_and_rejections(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pilot_labels, "decisions", {
        "aaaaaaaa": ("continue", "Safe"), "bbbbbbbb": ("block", "Unsafe")})
    queue = tmp_path / "queue.jsonl"
    output = tmp_path / "labels.jsonl"
    jsonl(queue, [{"event_uuid": "aaaaaaaa-1"}, {"event_uuid": "bbbbbbbb-2"}])
    invoke(monkeypatch, pilot_labels, "--queue", queue, "--output", output)
    assert json.loads(capsys.readouterr().out)["labels"] == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["primary_action"] for row in rows] == ["continue", "block"]
    assert all(row["review_status"] == "provisional" for row in rows)
    jsonl(queue, [{"event_uuid": "aaaaaaaa-1"}])
    with pytest.raises(ValueError, match="count"):
        invoke(monkeypatch, pilot_labels, "--queue", queue, "--output", output)
    jsonl(queue, [{"event_uuid": "aaaaaaaa-1"}, {"event_uuid": "aaaaaaaa-2"}])
    with pytest.raises(ValueError, match="Missing or duplicate"):
        invoke(monkeypatch, pilot_labels, "--queue", queue, "--output", output)


def test_teacher_second_pass_success_and_guardrails(tmp_path, monkeypatch, capsys):
    proposals = [
        {"event_uuid": "aaaaaaaa-1", "task_key": "task", "split": "train"},
        {"event_uuid": "bbbbbbbb-2", "task_key": "task", "split": "validation"},
    ]
    monkeypatch.setattr(adjudicate, "checkpoints", lambda *args: iter(proposals))
    monkeypatch.setattr(adjudicate, "CHANGES", {"aaaaaaaa": ("block", "Unsafe")})
    monkeypatch.setattr(adjudicate, "ADDITIONS", {"bbbbbbbb": ("query", "Unknown")})
    initial, output, log = (tmp_path / name for name in ("initial.jsonl", "out.jsonl", "log.json"))
    jsonl(initial, [{"event_uuid": "aaaaaaaa-1", "expected_actions": ["continue"],
                     "primary_action": "continue"}])
    args = ("--tasks", tmp_path / "tasks", "--events-dir", tmp_path / "events",
            "--initial-labels", initial, "--output", output, "--review-log", log)
    invoke(monkeypatch, adjudicate, *args)
    assert json.loads(capsys.readouterr().out)["labels"] == 2
    assert json.loads(log.read_text())["independent_human_review"] is False
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["primary_action"] for row in rows] == ["block", "query"]
    monkeypatch.setattr(adjudicate, "CHANGES", {"missing1": ("block", "Unsafe")})
    with pytest.raises(ValueError, match="absent"):
        invoke(monkeypatch, adjudicate, *args)
    monkeypatch.setattr(adjudicate, "CHANGES", {"bbbbbbbb": ("block", "Unsafe")})
    monkeypatch.setattr(adjudicate, "ADDITIONS", {})
    with pytest.raises(ValueError, match="No initial label"):
        invoke(monkeypatch, adjudicate, *args)
    monkeypatch.setattr(adjudicate, "CHANGES", {})
    monkeypatch.setattr(adjudicate, "ADDITIONS", {"aaaaaaaa": ("query", "Unknown")})
    with pytest.raises(ValueError, match="already exists"):
        invoke(monkeypatch, adjudicate, *args)
    jsonl(initial, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="Duplicate initial"):
        invoke(monkeypatch, adjudicate, *args)
    monkeypatch.setattr(adjudicate, "checkpoints", lambda *args: iter([proposals[0],
                            {**proposals[1], "event_uuid": "aaaaaaaa-2"}]))
    with pytest.raises(ValueError, match="Ambiguous"):
        invoke(monkeypatch, adjudicate, *args)


def test_inspect_queue_formats_selected_prefix(tmp_path, monkeypatch, capsys):
    queue = tmp_path / "queue.jsonl"
    jsonl(queue, [{"event_uuid": "abcd-1", "split": "train", "task_key": "task",
                   "checkpoint": {"sequence_number": 4, "agent_id": "agent1",
                       "task": {"goals": [{"owner_agent_id": "agent1", "description": "goal"}]},
                       "prior_events": [{"sequence_number": 2, "actor_id": "agent2",
                                         "event_type": "proposed_action",
                                         "payload": {"action": {"command": "git status"}}},
                                        {"sequence_number": 3, "actor_id": None,
                                         "event_type": "action_result",
                                         "payload": {"output_excerpt": "ok"}}],
                       "proposal": {"action": {"command": "git diff"}}}}])
    invoke(monkeypatch, inspect_queue, queue, "nomatch")
    assert capsys.readouterr().out == ""
    invoke(monkeypatch, inspect_queue, queue, "abcd")
    printed = capsys.readouterr().out
    assert "GOAL agent1: goal" in printed and "PRIOR 2" in printed and "PROPOSAL: git diff" in printed
    assert inspect_queue.short({"x": "a b"}, 100).startswith('{"x"')
    assert inspect_queue.short("a  b\nc", 3) == "a b..."
    original = json.loads(queue.read_text(encoding="utf-8").splitlines()[0])
    original["observation"] = {
        "proposal_sequence": 4, "actor_id": "agent1", "shared_goal": "Ship both features",
        "agent_goals": {"agent1": "goal"}, "constraints": {},
        "repository_state": {"pending_proposals": {"agent2": "queued"}},
        "recent_events": [{"sequence_number": 2, "actor_id": "agent2",
                           "event_type": "proposed_action", "payload": {"action": "git status"}}],
        "proposed_action": {"action": {"tool": "bash", "command": "git diff"}},
    }
    jsonl(queue, [original])
    invoke(monkeypatch, inspect_queue, queue, "abcd")
    printed = capsys.readouterr().out
    assert "REPOSITORY STATE:" in printed and "pending_proposals" in printed
    assert "SHARED GOAL: Ship both features" in printed and "PRIOR 2" in printed


def test_trace_action_parser_handles_missing_and_duplicate_calls():
    trace = {"messages": [
        {"role": "assistant", "extra": {"timestamp": 10}, "tool_calls": [
            {"id": "one", "function": {"name": "bash", "arguments": '{"command":"git status"}'}},
            {"id": "one", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "two", "function": {"arguments": "bad-json"}},
            {"id": "three", "function": {"arguments": "[]"}},
            {"id": None},
        ]},
        {"role": "tool", "tool_call_id": "one", "extra": {"timestamp": 11, "returncode": 0},
         "content": "token=secret"},
        {"role": "assistant", "tool_calls": [{"id": "no-time"}]},
        {"role": "user", "content": "ignore"},
    ]}
    actions = import_traces.trace_actions(trace, "agent1")
    assert len(actions) == 3
    assert actions[0]["command"] == "git status"
    assert actions[0]["result"] == "token=[REDACTED]"
    assert "'raw': 'bad-json'" in actions[1]["command"]
    assert "'raw': '[]'" in actions[2]["command"]
    assert import_traces.trace_actions({}, "agent2") == []
    assert import_traces.redact("password: secret", 9) == "password:"
    assert import_traces.iso_time(0).startswith("1970-01-01T00:00:00.000")


def test_trace_bundle_builds_ordered_events(tmp_path, monkeypatch):
    row = {"repo": "sample", "task_id": 1, "f1": 1, "f2": 2,
           "task_key": f"cooperbench:sample:task1:f1-f2:{import_traces.TASK_REVISION}", "split": "train"}
    trace = {"messages": [
        {"role": "assistant", "extra": {"timestamp": 10}, "tool_calls": [
            {"id": "call", "function": {"name": "bash", "arguments": '{"command":"git status"}'}}]},
        {"role": "tool", "tool_call_id": "call", "extra": {"timestamp": 11, "returncode": 0},
         "content": "ok"},
    ]}
    def download(repo, revision, filename):
        return json.dumps(trace) if filename.endswith("full_traj.json") else "Feature specification"
    monkeypatch.setattr(import_traces, "download_text", download)
    task, count = import_traces.make_bundle(row, {"agent1": "agent1_full_traj.json",
                                               "agent2": "agent2_full_traj.json"}, tmp_path)
    assert count == 2 and task["goals"][0]["description"] == "Feature specification"
    events = [json.loads(line) for line in next(tmp_path.rglob("events.jsonl")).read_text().splitlines()]
    assert events[0]["event_type"] == "run_start"
    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))
    assert events[-1]["trigger_sequence"] in {2, 3}
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())
    assert manifest["event_count"] == len(events)
    monkeypatch.setattr(import_traces, "download_text", lambda *args: json.dumps({"messages": []}))
    with pytest.raises(ValueError, match="No tool proposals"):
        import_traces.make_bundle(row, {"agent1": "agent1_full_traj.json",
                                        "agent2": "agent2_full_traj.json"}, tmp_path)
    invalid_trace = deepcopy(trace)
    invalid_trace["messages"][0]["tool_calls"][0]["function"]["name"] = "terminal"
    monkeypatch.setattr(import_traces, "download_text",
                        lambda *args: json.dumps(invalid_trace) if args[-1].endswith("full_traj.json")
                        else "Feature specification")
    with pytest.raises(ValueError, match="unsupported by the runner"):
        import_traces.make_bundle(row, {"agent1": "agent1_full_traj.json",
                                        "agent2": "agent2_full_traj.json"}, tmp_path)


def test_trace_bundle_keeps_proposal_without_tool_result(tmp_path, monkeypatch):
    row = {"repo": "sample", "task_id": 1, "f1": 1, "f2": 2,
           "task_key": f"cooperbench:sample:task1:f1-f2:{import_traces.TASK_REVISION}", "split": "train"}
    trace = {"messages": [{"role": "assistant", "extra": {"timestamp": 10},
             "tool_calls": [{"id": "call", "function": {"name": "bash",
                            "arguments": '{"command":"git status"}'}}]}]}
    monkeypatch.setattr(import_traces, "download_text",
                        lambda *args: json.dumps(trace) if args[-1].endswith(".json") else "Feature")
    _, count = import_traces.make_bundle(row, {"agent1": "agent1_full_traj.json",
                                            "agent2": "agent2_full_traj.json"}, tmp_path)
    assert count == 2
    events = [json.loads(line) for line in next(tmp_path.rglob("events.jsonl")).read_text().splitlines()]
    assert [item["event_type"] for item in events] == ["run_start", "proposed_action", "proposed_action"]


def test_trace_import_main_selects_complete_pairs(tmp_path, monkeypatch, capsys):
    row = {"repo": "sample", "task_id": 1, "f1": 1, "f2": 2,
           "task_key": f"cooperbench:sample:task1:f1-f2:{import_traces.TASK_REVISION}", "split": "train"}
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"revision": import_traces.TASK_REVISION, "pairs": [row]}))
    prefix = import_traces.TRACE_PREFIX
    paths = [f"{prefix}sample/1/f1_f2/{actor}_full_traj.json" for actor in ("agent1", "agent2")]
    paths += ["unrelated/path", f"{prefix}sample/1/f1_f2/other.txt",
              f"{prefix}sample/1/f1_f2/extra/agent1_full_traj.json"]
    monkeypatch.setattr(import_traces, "HfApi", lambda: SimpleNamespace(list_repo_files=lambda *a, **kw: paths))
    monkeypatch.setattr(import_traces, "make_bundle", lambda row, candidate, out: ({"task_key": row["task_key"]}, 2))
    out = tmp_path / "out"
    args = ("--split-manifest", manifest, "--output-dir", out, "--splits", "train", "--max-pairs-per-split", "1")
    invoke(monkeypatch, import_traces, *args)
    assert json.loads(capsys.readouterr().out)["proposals"] == 2
    assert json.loads((out / "provenance.json").read_text())["imported"][0]["trace_paths"]
    manifest.write_text(json.dumps({"revision": "wrong", "pairs": [row]}))
    with pytest.raises(ValueError, match="revision differs"):
        invoke(monkeypatch, import_traces, *args)
    manifest.write_text(json.dumps({"revision": import_traces.TASK_REVISION,
                                    "pairs": [{**row, "task_key": "legacy"}]}))
    with pytest.raises(ValueError, match="A2 normalized identity"):
        invoke(monkeypatch, import_traces, *args)
    manifest.write_text(json.dumps({"revision": import_traces.TASK_REVISION, "pairs": [row]}))
    monkeypatch.setattr(import_traces, "HfApi", lambda: SimpleNamespace(list_repo_files=lambda *a, **kw: []))
    with pytest.raises(ValueError, match="No complete"):
        invoke(monkeypatch, import_traces, *args)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, import_traces, *args, "--max-pairs-per-split", "0")


def test_trace_import_round_robins_and_skips_other_split(tmp_path, monkeypatch, capsys):
    def row(repo, number, split):
        return {"repo": repo, "task_id": number, "f1": 1, "f2": 2,
                "task_key": f"cooperbench:{repo}:task{number}:f1-f2:{import_traces.TASK_REVISION}",
                "split": split}
    rows = [row("alpha", 1, "train"), row("beta", 1, "train"),
            row("beta", 2, "train"), row("gamma", 1, "validation")]
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"revision": import_traces.TASK_REVISION, "pairs": rows}))
    prefix = import_traces.TRACE_PREFIX
    paths = [f"{prefix}{r['repo']}/{r['task_id']}/f1_f2/{actor}_full_traj.json"
             for r in rows for actor in ("agent1", "agent2")]
    monkeypatch.setattr(import_traces, "HfApi", lambda: SimpleNamespace(list_repo_files=lambda *a, **kw: paths))
    monkeypatch.setattr(import_traces, "make_bundle",
                        lambda r, candidate, out: ({"task_key": r["task_key"]}, 1))
    out = tmp_path / "out"
    base = ("--split-manifest", manifest, "--output-dir", out, "--splits", "train")
    invoke(monkeypatch, import_traces, *base, "--max-pairs-per-split", "3")
    assert json.loads(capsys.readouterr().out)["pairs"] == 3
    selected = [json.loads(line)["task_key"] for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert selected == [rows[0]["task_key"], rows[1]["task_key"], rows[2]["task_key"]]
    invoke(monkeypatch, import_traces, *base, "--max-pairs-per-split", "1")
    assert json.loads(capsys.readouterr().out)["pairs"] == 1


def test_pinned_download_helpers_use_passed_revision(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text('{"tasks": []}', encoding="utf-8")
    calls = []
    def fake_download(**kwargs):
        calls.append(kwargs)
        return source
    monkeypatch.setattr(import_traces, "hf_hub_download", fake_download)
    monkeypatch.setattr(split_cooperbench, "hf_hub_download", fake_download)
    assert import_traces.download_text("repo", "revision", "file") == '{"tasks": []}'
    assert split_cooperbench.source_json("file") == {"tasks": []}
    assert calls[0]["revision"] == "revision"
    assert calls[1]["revision"] == split_cooperbench.REVISION
