"""Offline tests for the new runner core, discovered by existing A2 CI."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from runner.event_writer import EventWriter
from scripts.run_artifacts import verify_bundle, load_events


class RunnerCoreTests(unittest.TestCase):
    def writer(self):
        return EventWriter(task_key="fixture", arbiter_condition="llm",
                           agent_config={}, arbiter_config={"training": "none"}, seed=42)

    def proposal(self, writer):
        return writer.emit("agent", "proposed_action", {
            "action": {"tool": "bash", "command": "echo hello"},
            "files_touched": [],
        }, actor_id="agent1")

    def test_bundle_compatible_with_a2(self):
        writer = self.writer()
        proposal = self.proposal(writer)
        writer.emit("arbiter", "arbiter_decision", {
            "action": "block", "reason": "fixture", "affected_agents": ["agent1"], "evidence": {},
        }, actor_id="arbiter", trigger_sequence=proposal)
        writer.emit("system", "action_result", {"outcome": "cancelled"},
                    trigger_sequence=proposal)
        writer.finish("completed")
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            writer.write_bundle(path / "events.jsonl", path / "bundle")
            manifest = verify_bundle(path / "bundle")
            events, _ = load_events((path / "bundle" / "events.jsonl").read_bytes())
            self.assertTrue(manifest["complete"])
            self.assertEqual([e["sequence_number"] for e in events], list(range(1, 6)))
            self.assertEqual(events[2]["trigger_sequence"], proposal)
            with self.assertRaises(FileExistsError):
                writer.write_bundle(path / "events.jsonl", path / "bundle")

    def test_reject_wrong_trigger_and_duplicate_result(self):
        writer = self.writer()
        with self.assertRaises(ValueError):
            writer.emit("system", "action_result", {"outcome": "cancelled"}, trigger_sequence=1)
        proposal = self.proposal(writer)
        writer.emit("system", "action_result", {"outcome": "cancelled"}, trigger_sequence=proposal)
        with self.assertRaises(ValueError):
            writer.emit("system", "action_result", {"outcome": "cancelled"}, trigger_sequence=proposal)

    def test_snapshot_is_independent(self):
        writer = self.writer()
        payload = {"message": "original"}
        writer.emit("agent", "message", payload, actor_id="agent1")
        payload["message"] = "changed"
        snapshot = writer.snapshot()
        snapshot[1]["payload"]["message"] = "also changed"
        self.assertEqual(writer.snapshot()[1]["payload"]["message"], "original")

    def test_no_events_after_end(self):
        writer = self.writer()
        writer.finish("completed")
        with self.assertRaises(ValueError):
            self.proposal(writer)

    def test_incomplete_bundle(self):
        writer = self.writer()
        self.proposal(writer)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            writer.write_bundle(path / "events.jsonl", path / "bundle")
            self.assertFalse(verify_bundle(path / "bundle")["complete"])
