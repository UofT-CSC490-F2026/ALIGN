"""Exercise training preparation against an actual merged runner event bundle."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("runner")
sys.path.insert(0, str(Path("scripts").resolve()))

from arbiter_training import prepare, synthetic
from arbiter_training.inference import ModalArbiter
from runner.arbitration import ArbitrationGate
from runner.cli import main as run_cli
from runner.concurrent import run_concurrent
from runner.environment import MockEnvironment
from runner.event_writer import EventWriter
from runner.orchestrator import validate_proposal
from scripts.normalize_tasks import normalized_custom, write_output


def test_runner_bundle_keeps_exact_decision_time_observation(tmp_path):
    tasks_dir = tmp_path / "tasks"
    write_output(normalized_custom(Path("scripts/examples"), "v1"), tasks_dir,
                 "custom", "v1", "all")
    assert run_cli(["--tasks", str(tasks_dir / "tasks.jsonl"),
                    "--output-dir", str(tmp_path / "runs"),
                    "--condition", "llm", "--mock-action", "block"]) == 0
    bundle = next((tmp_path / "runs").glob("*/bundle/events.jsonl"))
    events = list(prepare.read_jsonl(bundle))
    proposal = next(item for item in events if item["event_type"] == "proposed_action")
    recorded = next(item["payload"] for item in events
                    if item["event_type"] == "arbiter_observation"
                    and item.get("trigger_sequence") == proposal["sequence_number"])
    assert recorded["repository_state"]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"event_uuid": proposal["event_uuid"],
                                  "expected_actions": ["block"],
                                  "reviewer": "integration-test", "rationale": "Fixture block"}) + "\n",
                      encoding="utf-8")
    rows = prepare.build_records(tasks_dir / "tasks.jsonl", tmp_path / "runs", labels)
    assert len(rows) == 1
    assert rows[0]["observation_source"] == "runner_event"
    assert json.loads(rows[0]["prompt"][1]["content"]) == recorded


def test_all_synthetic_proposals_match_runner_tool_contract():
    task = {"task_key": "cooperbench:fixture:task1:f1-f2:revision",
            "shared_goal": "Both goals", "goals": [
                {"owner_agent_id": "agent1", "description": "First"},
                {"owner_agent_id": "agent2", "description": "Second"}],
            "metadata": {"split": "train"}}
    for number, case in enumerate(synthetic.CASES):
        row = synthetic.build_record(number, case, task)
        observation = json.loads(row["prompt"][1]["content"])
        assert validate_proposal(observation["proposed_action"]) == observation["proposed_action"]


def test_modal_adapter_decision_passes_merged_runner_gate():
    task = normalized_custom(Path("scripts/examples"), "v1")[0]

    class Agent:
        def __init__(self, proposal=None):
            self.proposal_once = proposal

        def propose(self, context):
            result, self.proposal_once = self.proposal_once, None
            return result

        def receive(self, result):
            pass

    class Remote:
        def __init__(self):
            self.decide_batch = self

        def remote(self, prompts):
            observation = json.loads(prompts[0][1]["content"])
            return [{"text": json.dumps({"action": "block", "reason": "Fixture block",
                                         "affected_agents": [observation["actor_id"]],
                                         "evidence": {"proposal_sequence":
                                                      observation["proposal_sequence"]}})}]

    arbiter = ModalArbiter(remote_model=Remote())
    writer = EventWriter(task_key=task["task_key"], arbiter_condition="llm",
                         agent_config={}, arbiter_config={})
    agents = {"agent1": Agent({"action": {"tool": "bash", "command": "echo fixture"},
                                "files_touched": []}), "agent2": Agent()}
    environment = MockEnvironment()
    events = run_concurrent(task, agents, environment, writer,
                            gate=ArbitrationGate("llm", arbiter))
    assert any(item["event_type"] == "arbiter_decision"
               and item["payload"]["action"] == "block" for item in events)
    assert not environment.executed
