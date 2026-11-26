"""Offline regression tests for Person 2's decision data and runner interface."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")
pytest.importorskip("huggingface_hub")

from arbiter_training import contract, prepare, merge_review, synthetic
from arbiter_training import audit, inference, review_packet, review_queue
from arbiter_training import review_sheet, prepare_test, split_cooperbench


def task(repo="sample", split="train"):
    return {
        "task_key": f"cooperbench:{repo}:task1:f1_f2",
        "shared_goal": "Implement both features",
        "constraints": {},
        "goals": [
            {"goal_key": "feature1", "owner_agent_id": "agent1", "description": "First"},
            {"goal_key": "feature2", "owner_agent_id": "agent2", "description": "Second"},
        ],
        "metadata": {"split": split},
    }


def event(kind, sequence, payload, actor="agent1", uuid=None):
    return {
        "run_key": "run-1", "sequence_number": sequence,
        "event_uuid": uuid or f"event-{sequence}", "event_type": kind,
        "actor_type": "agent" if actor else "system", "actor_id": actor,
        "payload": payload,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_files(tmp_path, split="train"):
    t = task(split=split)
    tasks = tmp_path / "tasks.jsonl"
    events_dir = tmp_path / "events"
    labels = tmp_path / "labels.jsonl"
    write_jsonl(tasks, [t])
    write_jsonl(events_dir / "one" / "events.jsonl", [
        event("run_start", 1, {"task_key": t["task_key"]}, actor=None),
        event("proposed_action", 2, {
            "action": {"tool": "bash", "command": "git status"}, "files_touched": []}),
        event("action_result", 3, {"outcome": "dispatched"}, actor=None),
    ])
    write_jsonl(labels, [{"event_uuid": "event-2", "expected_actions": ["continue"],
                          "primary_action": "continue", "reviewer": "Jinhao",
                          "rationale": "Read only", "label_source": "human"}])
    return tasks, events_dir, labels


def run_main(monkeypatch, module, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, args)])
    module.main()


def test_contract_round_trip_and_schema_rejection():
    checkpoint = {"schema_version": 1, "task": task(), "agent_id": "agent1",
                  "sequence_number": 4, "proposal": {
                      "action": {"tool": "bash", "command": "git status"},
                      "files_touched": []},
                  "prior_events": [{"sequence_number": 1, "payload": {}}]}
    observation = contract.observation_from_checkpoint(checkpoint)
    assert observation["agent_goals"] == {"agent1": "First", "agent2": "Second"}
    assert observation["repository_state"] == {}
    assert contract.messages_for(checkpoint) == contract.messages_for_observation(observation)
    decision = contract.decision_for_observation("query", "Inspect state", observation)
    assert decision["affected_agents"] == ["agent1"]
    assert decision["evidence"]["recent_event_sequences"] == [1]
    assert contract.decision_for_observation("continue", "Safe", observation)["affected_agents"] == []
    assert contract.decision_for_observation("mediate", "Coordinate", observation)["affected_agents"] == ["agent1", "agent2"]
    assert contract.parse_decision("```json\n" + json.dumps(decision) + "\n```") == decision
    assert contract.parse_decision("```\n" + json.dumps(decision) + "\n```") == decision
    with pytest.raises(ValueError, match="checkpoint schema"):
        contract.observation_from_checkpoint({**checkpoint, "schema_version": 2})
    with pytest.raises(ValueError, match="Observation"):
        contract.messages_for_observation({**observation, "extra": True})
    with pytest.raises(ValueError, match="ProposedAction"):
        contract.messages_for_observation({**observation, "proposed_action": {"action": {}}})


@pytest.mark.parametrize("raw,message", [
    ("not-json", "not valid JSON"),
    ('{"action":"continue"}', "four decision fields"),
    (json.dumps({"action": "unsafe", "reason": "r", "affected_agents": [], "evidence": {}}), "invalid action"),
    (json.dumps({"action": "continue", "reason": " ", "affected_agents": [], "evidence": {}}), "invalid action"),
    (json.dumps({"action": "continue", "reason": "r", "affected_agents": [1], "evidence": {}}), "invalid affected"),
])
def test_contract_rejects_bad_decisions(raw, message):
    with pytest.raises(ValueError, match=message):
        contract.parse_decision(raw)


def test_decision_matches_runner_agent_ids_and_enforcement():
    observation = contract.observation_from_checkpoint({
        "schema_version": 1, "task": task(), "agent_id": "agent1",
        "sequence_number": 2, "proposal": {"action": {"tool": "bash", "command": "git status"},
                                           "files_touched": []}, "prior_events": []})
    decision = contract.decision_for_observation("query", "Check", observation)
    assert contract.parse_decision(json.dumps(decision), observation, True) == decision
    bad_agent = {**decision, "affected_agents": ["unknown"]}
    with pytest.raises(ValueError, match="unknown affected agent"):
        contract.parse_decision(json.dumps(bad_agent), observation, True)
    reserved = {**decision, "action": "restructure"}
    assert contract.parse_decision(json.dumps(reserved))["action"] == "restructure"
    with pytest.raises(ValueError, match="unsupported enforced action"):
        contract.parse_decision(json.dumps(reserved), observation, True)
    nonfinite = {**decision, "evidence": {"score": float("nan")}}
    with pytest.raises(ValueError, match="JSON serializable"):
        contract.parse_decision(json.dumps(nonfinite), observation, True)


def test_prepare_records_and_split(tmp_path):
    tasks, events_dir, labels = fixture_files(tmp_path)
    rows = prepare.build_records(tasks, events_dir, labels)
    assert len(rows) == 1
    row = rows[0]
    assert row["label_source"] == "human"
    assert row["review_status"] == "reviewed"
    assert row["primary_action"] == "continue"
    assert row["observation_source"] == "public_reconstruction"
    assert json.loads(row["prompt"][1]["content"])["recent_events"][0]["event_type"] == "run_start"
    assert json.loads(row["completion"][0]["content"])["action"] == "continue"
    assert prepare.split_records(rows, .2) == (rows, [])
    assert prepare.split_records([{**row, "split": "validation"}], .2) == ([], [{**row, "split": "validation"}])
    with pytest.raises(ValueError, match="validation_fraction"):
        prepare.split_records(rows, 1)
    assert prepare.build_records(tasks, events_dir, labels, excluded_tasks={task()["task_key"]}) == []


def test_prepare_uses_recorded_runner_observation(tmp_path):
    tasks, events_dir, labels = fixture_files(tmp_path)
    path = events_dir / "one" / "events.jsonl"
    rows = list(prepare.read_jsonl(path))
    checkpoint = {"schema_version": 1, "task": task(), "agent_id": "agent1",
                  "sequence_number": 2, "proposal": rows[1]["payload"],
                  "prior_events": []}
    recorded = contract.observation_from_checkpoint(checkpoint)
    recorded["repository_state"] = {"pending_proposals": {"agent2": {"phase": "queued"}},
                                    "in_flight_actions": {"agent2": {"started": True}}}
    recorded["recent_events"] = []
    rows.insert(2, {**event("arbiter_observation", 3, recorded, actor=None),
                    "trigger_sequence": 2})
    rows[3]["sequence_number"] = 4
    write_jsonl(path, rows)
    result = prepare.build_records(tasks, events_dir, labels)[0]
    assert result["observation_source"] == "runner_event"
    assert json.loads(result["prompt"][1]["content"]) == recorded
    write_jsonl(path, [*rows, {**rows[2], "sequence_number": 5}])
    with pytest.raises(ValueError, match="Invalid arbiter_observation link"):
        prepare.build_records(tasks, events_dir, labels)
    write_jsonl(path, [rows[0], rows[1], {**rows[2], "payload": {**recorded, "actor_id": "agent2"}}, rows[3]])
    with pytest.raises(ValueError, match="disagrees with proposal"):
        prepare.build_records(tasks, events_dir, labels)
    second = event("proposed_action", 5, rows[1]["payload"], uuid="event-5")
    write_jsonl(path, [*rows, second])
    first_label = list(prepare.read_jsonl(labels))[0]
    write_jsonl(labels, [first_label, {**first_label, "event_uuid": "event-5"}])
    with pytest.raises(ValueError, match="Missing arbiter_observation"):
        prepare.build_records(tasks, events_dir, labels)


def test_prepare_uses_staged_copy_once_and_rejects_divergence(tmp_path):
    tasks, events_dir, labels = fixture_files(tmp_path)
    raw = events_dir / "one" / "events.jsonl"
    staged = events_dir / "one" / "bundle" / "events.jsonl"
    staged.parent.mkdir()
    staged.write_bytes(raw.read_bytes())
    rows = prepare.build_records(tasks, events_dir, labels)
    assert len(rows) == 1 and rows[0]["source_bundle"] == str(staged)
    staged.write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="Raw events differ"):
        prepare.build_records(tasks, events_dir, labels)


def test_prepare_rejects_bad_inputs(tmp_path):
    tasks, events_dir, labels = fixture_files(tmp_path)
    with pytest.raises(ValueError, match="No events"):
        prepare.build_records(tasks, tmp_path / "missing", labels)
    write_jsonl(tasks, [task(), task()])
    with pytest.raises(ValueError, match="Duplicate task_key"):
        prepare.build_records(tasks, events_dir, labels)
    write_jsonl(tasks, [task()])
    write_jsonl(labels, [{"event_uuid": "missing", "expected_actions": ["block"],
                          "reviewer": "J", "rationale": "Unsafe"}])
    with pytest.raises(ValueError, match="Labels without matching"):
        prepare.build_records(tasks, events_dir, labels)
    write_jsonl(labels, [{"event_uuid": "event-2", "expected_actions": ["block", "block"],
                          "reviewer": "J", "rationale": "Unsafe"}])
    with pytest.raises(ValueError, match="Invalid actions"):
        prepare.validated_labels(labels)


@pytest.mark.parametrize("field,value,message", [
    ("expected_actions", [], "nonempty"), ("expected_actions", ["invalid"], "Invalid actions"),
    ("reviewer", "", "needs a reviewer"), ("rationale", "", "needs a rationale"),
    ("primary_action", "block", "acceptable"), ("label_source", "fake", "provenance"),
])
def test_label_validation(tmp_path, field, value, message):
    row = {"event_uuid": "e", "expected_actions": ["continue"],
           "reviewer": "J", "rationale": "Safe"}
    row[field] = value
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [row])
    with pytest.raises(ValueError, match=message):
        prepare.validated_labels(labels)


def test_merge_review_changes_only_reviewed_rows(tmp_path, monkeypatch, capsys):
    teacher = [{"event_uuid": "one", "expected_actions": ["continue"],
                "primary_action": "continue"},
               {"event_uuid": "two", "expected_actions": ["query"],
                "primary_action": "query"}]
    review = [{"event_uuid": "one", "expected_actions": ["block"],
               "primary_action": "block", "reviewer": "Jinhao", "rationale": "Would overwrite"},
              {"event_uuid": "two"}]
    merged, summary = merge_review.merge(teacher, review)
    assert summary == {"reviewed": 1, "changed_primary_actions": 1, "remaining_provisional": 1}
    assert merged[0]["label_source"] == "human"
    assert merged[1] == teacher[1]
    for bad in ([{**review[0], "event_uuid": "missing"}], [review[0], review[0]],
                [{"event_uuid": "one", "expected_actions": ["block"]}]):
        with pytest.raises(ValueError):
            merge_review.merge(teacher, bad)
    with pytest.raises(ValueError, match="Duplicate teacher"):
        merge_review.merge([teacher[0], teacher[0]], review)
    with pytest.raises(ValueError, match="No completed"):
        merge_review.merge(teacher, [review[1]])
    teacher_path, review_path, output = (tmp_path / name for name in ("teacher.jsonl", "review.jsonl", "out.jsonl"))
    write_jsonl(teacher_path, teacher)
    write_jsonl(review_path, review)
    run_main(monkeypatch, merge_review, "--teacher-labels", teacher_path,
             "--reviewed-packet", review_path, "--output", output)
    assert json.loads(capsys.readouterr().out)["reviewed"] == 1
    assert list(prepare.read_jsonl(output)) == merged


class FakeRemote:
    def __init__(self, outputs):
        self.outputs = outputs
        self.decide_batch = self
        self.prompts = None

    def remote(self, prompts):
        self.prompts = prompts
        return self.outputs


def test_modal_arbiter_offline_interface():
    observation = contract.observation_from_checkpoint({
        "schema_version": 1, "task": task(), "agent_id": "agent1",
        "sequence_number": 3, "proposal": {"action": {"tool": "bash", "command": "git status"},
                                           "files_touched": []}, "prior_events": []})
    expected = contract.decision_for_observation("continue", "Safe", observation)
    remote = FakeRemote([{"text": json.dumps(expected)}])
    arbiter = inference.ModalArbiter(remote_model=remote)
    assert arbiter.decide(observation) == expected
    assert remote.prompts == [contract.messages_for_observation(observation)]
    assert arbiter.decide_many([]) == []
    remote.outputs = []
    with pytest.raises(RuntimeError, match="different number"):
        arbiter.decide(observation)
    remote.outputs = [{"text": "bad JSON"}]
    with pytest.raises(ValueError):
        arbiter.decide(observation)
    fallback = inference.ModalArbiter(remote_model=remote, invalid_policy="query")
    assert fallback.decide(observation)["evidence"]["error"] == "invalid_model_output"
    remote.outputs = [{"text": json.dumps({**expected, "affected_agents": ["missing"]})}]
    assert fallback.decide(observation)["action"] == "query"
    remote.outputs = [{"text": json.dumps({**expected, "action": "restructure"})}]
    assert fallback.decide(observation)["action"] == "query"
    for kwargs in ({"model": "wrong"}, {"model": "gpt-oss", "adapter_path": "adapter"},
                   {"invalid_policy": "continue"}):
        with pytest.raises(ValueError):
            inference.ModalArbiter(remote_model=remote, **kwargs)


def test_synthetic_record_preserves_provenance(tmp_path, monkeypatch, capsys):
    case = ("sample", "Earlier observation", "git diff", "continue", "Read only")
    row = synthetic.build_record(0, case, task())
    assert row["scenario_source"] == "synthetic_counterfactual"
    assert row["review_status"] == "provisional"
    assert row["label_source"] == "teacher_gpt"
    assert json.loads(row["prompt"][1]["content"])["proposed_action"]["action"]["command"] == "git diff"
    assert json.loads(row["prompt"][1]["content"])["proposed_action"]["action"]["tool"] == "bash"
    tasks = tmp_path / "tasks.jsonl"
    prepared, output = tmp_path / "prepared", tmp_path / "output"
    write_jsonl(tasks, [task()])
    write_jsonl(prepared / "train.jsonl", [])
    write_jsonl(prepared / "validation.jsonl", [])
    monkeypatch.setattr(synthetic, "CASES", [case])
    run_main(monkeypatch, synthetic, "--tasks", tasks, "--prepared-dir", prepared,
             "--output-dir", output)
    assert json.loads(capsys.readouterr().out)["train"] == 1
    assert list(prepare.read_jsonl(output / "train.jsonl"))[0]["primary_action"] == "continue"
    assert json.loads((output / "manifest.json").read_text())["synthetic_cases"] == 1
    monkeypatch.setattr(synthetic, "CASES", [("missing", *case[1:])])
    with pytest.raises(ValueError, match="No imported task"):
        run_main(monkeypatch, synthetic, "--tasks", tasks, "--prepared-dir", prepared,
                 "--output-dir", output)


def test_synthetic_message_uses_structured_runner_tool():
    case = ("sample", "Agents need to compare edits",
            {"tool": "message", "recipient": "agent1", "content": "Compare edits"},
            "continue", "Safe coordination")
    row = synthetic.build_record(1, case, task())
    action = json.loads(row["prompt"][1]["content"])["proposed_action"]["action"]
    assert action == case[2]


@pytest.mark.parametrize("command,kind", [
    ("send_message agent2", "communication"), ("coop-send hello", "communication"),
    ("git merge other", "integration"), ("git reset --hard HEAD", "destructive"),
    ("apply_patch x", "possible_edit"), ("pytest -q", "test"),
    ("git status", "git"), ("ls", "inspection"),
])
def test_review_category(command, kind):
    assert review_queue.category(command) == kind


def test_review_checkpoint_and_worksheet(tmp_path, monkeypatch, capsys):
    tasks, events_dir, labels = fixture_files(tmp_path)
    item = next(review_queue.checkpoints(tasks, events_dir, 8))
    assert item["category"] == "git"
    assert item["checkpoint"]["sequence_number"] == 2
    queue = tmp_path / "queue.jsonl"
    run_main(monkeypatch, review_queue, "--tasks", tasks, "--events-dir", events_dir,
             "--output", queue, "--limit-per-split", "1")
    assert json.loads(capsys.readouterr().out)["selected"] == 1
    assert len(list(prepare.read_jsonl(queue))) == 1
    packet = tmp_path / "packet.jsonl"
    observation = contract.observation_from_checkpoint(item["checkpoint"])
    write_jsonl(packet, [{"event_uuid": "event-2", "task_key": task()["task_key"],
                          "split": "train", "checkpoint": observation,
                          "expected_actions": [], "primary_action": None, "rationale": ""}])
    assert review_sheet.excerpt("a\nb", 10) == "a b"
    assert review_sheet.excerpt("abcdef", 3) == "abc..."
    rendered = review_sheet.render(list(prepare.read_jsonl(packet)))
    assert "git status" in rendered and "Teacher labels and model predictions are hidden" in rendered
    sheet = tmp_path / "sheet.md"
    run_main(monkeypatch, review_sheet, "--packet", packet, "--output", sheet)
    assert sheet.read_text(encoding="utf-8") == rendered
    assert json.loads(capsys.readouterr().out)["review_rows"] == 1
    filled = list(prepare.read_jsonl(packet))
    filled[0]["rationale"] = "already reviewed"
    write_jsonl(packet, filled)
    with pytest.raises(ValueError, match="blind"):
        run_main(monkeypatch, review_sheet, "--packet", packet, "--output", sheet)


def test_review_packet_filters_and_omits_answers(tmp_path, monkeypatch, capsys):
    tasks, events_dir, labels = fixture_files(tmp_path)
    row = prepare.build_records(tasks, events_dir, labels)[0]
    synthetic_row = {**row, "checkpoint_event_uuid": "synthetic", "scenario_source": "synthetic_counterfactual"}
    prepared = tmp_path / "prepared"
    write_jsonl(prepared / "train.jsonl", [row, synthetic_row])
    write_jsonl(prepared / "validation.jsonl", [])
    output = tmp_path / "review.jsonl"
    run_main(monkeypatch, review_packet, "--prepared-dir", prepared, "--output", output)
    assert json.loads(capsys.readouterr().out)["review_rows"] == 1
    exported = list(prepare.read_jsonl(output))
    assert exported[0]["expected_actions"] == []
    assert "completion" not in exported[0]
    audit_file = tmp_path / "audit.json"
    audit_file.write_text(json.dumps({"review_candidates": [{"event_uuid": "synthetic"}]}))
    run_main(monkeypatch, review_packet, "--prepared-dir", prepared, "--output", output,
             "--audit", audit_file, "--include-synthetic")
    assert json.loads(capsys.readouterr().out)["review_rows"] == 1
    assert list(prepare.read_jsonl(output))[0]["scenario_source"] == "synthetic_counterfactual"


def test_heldout_preparation_requires_independent_reviews(tmp_path, monkeypatch, capsys):
    tasks, events_dir, labels = fixture_files(tmp_path, "test")
    output = tmp_path / "heldout.jsonl"
    run_main(monkeypatch, prepare_test, "--tasks", tasks, "--events-dir", events_dir,
             "--labels", labels, "--output", output)
    assert json.loads(capsys.readouterr().out)["test_records"] == 1
    assert list(prepare.read_jsonl(output))[0]["split"] == "test"
    write_jsonl(labels, [{"event_uuid": "event-2", "expected_actions": ["continue"],
                          "reviewer": None, "rationale": "Safe", "label_source": "teacher_gpt"}])
    with pytest.raises(ValueError, match="independently reviewed"):
        run_main(monkeypatch, prepare_test, "--tasks", tasks, "--events-dir", events_dir,
                 "--labels", labels, "--output", output)
    monkeypatch.setattr(prepare_test, "build_records", lambda *a, **k: [])
    with pytest.raises(ValueError, match="No labeled"):
        run_main(monkeypatch, prepare_test, "--tasks", tasks, "--events-dir", events_dir,
                 "--labels", labels, "--output", output)
    monkeypatch.setattr(prepare_test, "build_records", lambda *a, **k: [{"split": "train"}])
    with pytest.raises(ValueError, match="non-test"):
        run_main(monkeypatch, prepare_test, "--tasks", tasks, "--events-dir", events_dir,
                 "--labels", labels, "--output", output)


def test_split_cooperbench_keeps_repositories_apart(tmp_path, monkeypatch, capsys):
    source = {"tasks": [{"repo": "pallets_click_task", "task_id": 1, "pairs": [[2, 1]]},
                        {"repo": "sample", "task_id": 2, "pairs": [[3, 4]]}]}
    report = {"all_results": [
        {"repo": "pallets_click_task", "task_id": 1, "f1": 1, "f2": 2, "has_conflict": True},
        {"repo": "sample", "task_id": 2, "f1": 3, "f2": 4, "has_conflict": False}]}
    monkeypatch.setattr(split_cooperbench, "source_json", lambda name: source if name.startswith("subsets/") else report)
    rows = split_cooperbench.make_split("lite")
    assert [row["split"] for row in rows] == ["test", "train"]
    assert rows[0]["task_key"] == (
        f"cooperbench:pallets_click_task:task1:f1-f2:{split_cooperbench.REVISION}")
    output = tmp_path / "split.json"
    run_main(monkeypatch, split_cooperbench, "--output", output)
    assert json.loads(capsys.readouterr().out)["test"]["merge_conflicts"] == 1
    assert json.loads(output.read_text())["pairs"] == rows
    monkeypatch.setattr(split_cooperbench, "make_split", lambda subset: [
        {"repo": "same", "split": "train", "gold_patches_merge_conflict": False},
        {"repo": "same", "split": "validation", "gold_patches_merge_conflict": False}])
    with pytest.raises(ValueError, match="Repository leakage"):
        run_main(monkeypatch, split_cooperbench, "--output", output)


def test_audit_checks_provenance_and_future_leakage(tmp_path, monkeypatch, capsys):
    tasks, events_dir, labels = fixture_files(tmp_path)
    row = prepare.build_records(tasks, events_dir, labels)[0]
    prepared = tmp_path / "prepared"
    write_jsonl(prepared / "train.jsonl", [row])
    write_jsonl(prepared / "validation.jsonl", [])
    output = tmp_path / "audit.json"
    run_main(monkeypatch, audit, "--prepared-dir", prepared, "--output", output)
    assert json.loads(capsys.readouterr().out)["examples"]["train"] == 1
    assert json.loads(output.read_text())["warnings"] == ["Too few train interventions for a meaningful fine-tune"]
    bad = deepcopy(row)
    observation = json.loads(bad["prompt"][1]["content"])
    observation["recent_events"].append({"sequence_number": 2})
    bad["prompt"][1]["content"] = json.dumps(observation)
    write_jsonl(prepared / "train.jsonl", [bad])
    with pytest.raises(ValueError, match="Future event"):
        run_main(monkeypatch, audit, "--prepared-dir", prepared)


def test_prepare_cli_and_input_reading(tmp_path, monkeypatch, capsys):
    tasks, events_dir, labels = fixture_files(tmp_path)
    output = tmp_path / "prepared"
    args = ("--tasks", tasks, "--events-dir", events_dir, "--labels", labels,
            "--output-dir", output)
    run_main(monkeypatch, prepare, *args)
    assert json.loads(capsys.readouterr().out)["train"] == 1
    assert list(prepare.read_jsonl(output / "train.jsonl"))[0]["checkpoint_event_uuid"] == "event-2"
    excluded = tmp_path / "exclude.txt"
    excluded.write_text("# comment\n\n" + task()["task_key"] + "\n")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, prepare, *args, "--exclude-task-keys", excluded)
    with pytest.raises(SystemExit):
        run_main(monkeypatch, prepare, *args, "--history-limit", "-1")
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("\nnot json\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        list(prepare.read_jsonl(invalid))
    invalid.write_text("[]\n")
    with pytest.raises(ValueError, match="Expected an object"):
        list(prepare.read_jsonl(invalid))
    assert list(prepare.read_jsonl(output / "validation.jsonl")) == []


def test_prepare_rejects_invalid_bundles_and_duplicate_labels(tmp_path):
    tasks, events_dir, labels = fixture_files(tmp_path)
    original = list(prepare.read_jsonl(events_dir / "one" / "events.jsonl"))
    path = events_dir / "one" / "events.jsonl"
    variants = [
        (original[1:], "Missing run_start"),
        ([{**original[0], "payload": {"task_key": "missing"}}, *original[1:]], "Unknown task_key"),
        ([original[0], {**original[1], "sequence_number": 5}, original[2]], "event order"),
        ([original[0], original[1], {**original[1], "sequence_number": 3}], "Repeated labeled"),
    ]
    for rows, message in variants:
        write_jsonl(path, rows)
        with pytest.raises(ValueError, match=message):
            prepare.build_records(tasks, events_dir, labels)
    write_jsonl(path, original)
    row = list(prepare.read_jsonl(labels))[0]
    write_jsonl(labels, [row, row])
    with pytest.raises(ValueError, match="Duplicate label"):
        prepare.validated_labels(labels)
    write_jsonl(labels, [{**row, "label_source": "teacher_gpt", "reviewer": None}])
    assert prepare.validated_labels(labels)["event-2"][-1] == "provisional"
    assert prepare.split_records([{**row, "task_key": "hashed", "split": None}], .999999)[1]


@pytest.mark.parametrize("mutation,message", [
    (lambda row: row.update(split="validation"), "Record split"),
    (lambda row: row.update(label_source="teacher_gpt", review_status="reviewed"), "incorrectly marked"),
    (lambda row: row.update(primary_action="block"), "not in acceptable"),
    (lambda row: row.update(prompt=[]), "Unexpected prompt"),
    (lambda row: row["prompt"][1].update(content=json.dumps({"wrong": 1})), "observation schema"),
    (lambda row: row["completion"][0].update(content=json.dumps({"action": "block", "reason": "r", "affected_agents": [], "evidence": {"proposal_sequence": 2}})), "differs from label"),
    (lambda row: row["completion"][0].update(content=json.dumps({"action": "continue", "reason": "r", "affected_agents": [], "evidence": {"proposal_sequence": 999}})), "different proposal"),
])
def test_audit_rejects_corrupt_records(tmp_path, monkeypatch, mutation, message):
    tasks, events_dir, labels = fixture_files(tmp_path)
    row = prepare.build_records(tasks, events_dir, labels)[0]
    mutation(row)
    prepared = tmp_path / "prepared"
    write_jsonl(prepared / "train.jsonl", [row])
    write_jsonl(prepared / "validation.jsonl", [])
    with pytest.raises(ValueError, match=message):
        run_main(monkeypatch, audit, "--prepared-dir", prepared)


def test_audit_rejects_leakage_duplicates_and_public_state(tmp_path, monkeypatch):
    tasks, events_dir, labels = fixture_files(tmp_path)
    row = prepare.build_records(tasks, events_dir, labels)[0]
    prepared = tmp_path / "prepared"
    args = ("--prepared-dir", prepared)
    write_jsonl(prepared / "train.jsonl", [row])
    write_jsonl(prepared / "validation.jsonl", [{**row, "split": "validation"}])
    with pytest.raises(ValueError, match="Task leakage"):
        run_main(monkeypatch, audit, *args)
    other = deepcopy(row)
    other["split"] = "validation"
    other["task_key"] = "cooperbench:sample:task2:f1_f2"
    write_jsonl(prepared / "validation.jsonl", [other])
    with pytest.raises(ValueError, match="Repository leakage"):
        run_main(monkeypatch, audit, *args)
    write_jsonl(prepared / "validation.jsonl", [])
    write_jsonl(prepared / "train.jsonl", [row, row])
    with pytest.raises(ValueError, match="Duplicate checkpoint"):
        run_main(monkeypatch, audit, *args)
    public = deepcopy(row)
    public["scenario_source"] = "public_trace"
    obs = json.loads(public["prompt"][1]["content"])
    obs["repository_state"] = {"secret": "future"}
    public["prompt"][1]["content"] = json.dumps(obs)
    write_jsonl(prepared / "train.jsonl", [public])
    with pytest.raises(ValueError, match="no reliable repository snapshot"):
        run_main(monkeypatch, audit, *args)
    write_jsonl(prepared / "train.jsonl", [row])
    duplicate_prompt = deepcopy(row)
    duplicate_prompt["checkpoint_event_uuid"] = "new-event"
    write_jsonl(prepared / "train.jsonl", [row, duplicate_prompt])
    with pytest.raises(ValueError, match="Duplicate prompt"):
        run_main(monkeypatch, audit, *args)


def test_audit_flags_provisional_examples(tmp_path, monkeypatch, capsys):
    tasks, events_dir, labels = fixture_files(tmp_path)
    row = prepare.build_records(tasks, events_dir, labels)[0]
    row.update(label_source="teacher_gpt", review_status="provisional", primary_action="block",
               expected_actions=["block"])
    row["completion"][0]["content"] = json.dumps({"action": "block", "reason": "Unsafe",
        "affected_agents": ["agent1"], "evidence": {"proposal_sequence": 2}})
    prepared = tmp_path / "prepared"
    write_jsonl(prepared / "train.jsonl", [row])
    write_jsonl(prepared / "validation.jsonl", [])
    run_main(monkeypatch, audit, "--prepared-dir", prepared)
    result = json.loads(capsys.readouterr().out)
    assert result["review_candidates"][0]["action"] == "block"
    assert any("All labels are provisional" in message for message in result["warnings"])


def test_audit_enough_interventions_has_no_scarcity_warning(tmp_path, monkeypatch, capsys):
    tasks, events_dir, labels = fixture_files(tmp_path)
    template = prepare.build_records(tasks, events_dir, labels)[0]
    rows = []
    for number in range(5):
        row = deepcopy(template)
        row["checkpoint_event_uuid"] = f"block-{number}"
        row["primary_action"] = "block"
        row["expected_actions"] = ["block"]
        item = json.loads(row["prompt"][1]["content"])
        item["proposed_action"]["action"]["command"] = f"git reset --hard {number}"
        row["prompt"][1]["content"] = json.dumps(item)
        row["completion"][0]["content"] = json.dumps({"action": "block", "reason": "Unsafe",
            "affected_agents": ["agent1"], "evidence": {"proposal_sequence": 2}})
        rows.append(row)
    prepared = tmp_path / "prepared"
    write_jsonl(prepared / "train.jsonl", rows)
    write_jsonl(prepared / "validation.jsonl", [])
    run_main(monkeypatch, audit, "--prepared-dir", prepared)
    assert json.loads(capsys.readouterr().out)["warnings"] == []


def test_review_queue_selects_across_categories(tmp_path, monkeypatch, capsys):
    items = [{"event_uuid": "a", "task_key": "first", "split": "train", "category": "destructive"},
             {"event_uuid": "b", "task_key": "second", "split": "train", "category": "integration"}]
    monkeypatch.setattr(review_queue, "checkpoints", lambda *a: iter(items))
    output = tmp_path / "queue.jsonl"
    run_main(monkeypatch, review_queue, "--tasks", tmp_path / "tasks", "--events-dir",
             tmp_path / "events", "--output", output, "--train-limit", "2")
    assert json.loads(capsys.readouterr().out)["selected"] == 2
