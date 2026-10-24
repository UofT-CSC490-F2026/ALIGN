"""Offline tests for run bundle integrity and event contract."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_artifacts", SCRIPTS / "run_artifacts.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
EXAMPLE = SCRIPTS / "examples" / "example_run.events.jsonl"


class RunArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = json.loads("[" + ",".join(EXAMPLE.read_text().strip().splitlines()) + "]")

    def event_file(self, events):
        path = self.root / "input.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        return path

    def test_fixture_is_valid_and_complete(self):
        records, header = mod.load_events(EXAMPLE.read_bytes())
        self.assertEqual(len(records), 8)
        self.assertTrue(header["complete"])
        self.assertEqual(header["arbiter_condition"], "llm")

    def test_bundle_stage_verify_and_idempotency(self):
        dest = self.root / "bundle"
        original = mod.stage(EXAMPLE, dest)
        self.assertEqual(mod.verify_bundle(dest), original)
        self.assertEqual(mod.stage(EXAMPLE, dest), original)

    def test_bad_checksum_rejected(self):
        dest = self.root / "bundle"
        mod.stage(EXAMPLE, dest)
        (dest / "events.jsonl").write_bytes(b"tampered\n")
        with self.assertRaises(ValueError):
            mod.verify_bundle(dest)

    def test_sequence_gap_rejected(self):
        self.fixture[2]["sequence_number"] = 10
        with self.assertRaisesRegex(ValueError, "contiguous"):
            mod.load_events(self.event_file(self.fixture).read_bytes())

    def test_duplicate_uuid_rejected(self):
        self.fixture[2]["event_uuid"] = self.fixture[1]["event_uuid"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            mod.load_events(self.event_file(self.fixture).read_bytes())

    def test_forward_trigger_rejected(self):
        self.fixture[2]["trigger_sequence"] = 4
        with self.assertRaisesRegex(ValueError, "earlier"):
            mod.load_events(self.event_file(self.fixture).read_bytes())

    def test_partial_crashed_run_permitted(self):
        events = self.fixture[:-1]
        _, header = mod.load_events(self.event_file(events).read_bytes())
        self.assertFalse(header["complete"])

    def test_multiple_run_end_rejected(self):
        e = self.fixture[-1].copy()
        e["sequence_number"] = 9
        e["event_uuid"] = "12345678-1234-4234-8234-100000000099"
        self.fixture.append(e)
        with self.assertRaisesRegex(ValueError, "run_end"):
            mod.load_events(self.event_file(self.fixture).read_bytes())

    def test_missing_utc_rejected(self):
        self.fixture[0]["occurred_at"] = "2026-10-24T12:00:00"
        with self.assertRaisesRegex(ValueError, "UTC"):
            mod.load_events(self.event_file(self.fixture).read_bytes())

    def test_invalid_actor_and_unknown_fields_rejected(self):
        self.fixture[2]["actor_type"] = "agent"
        with self.assertRaisesRegex(ValueError, "Decision"):
            mod.load_events(self.event_file(self.fixture).read_bytes())
        self.fixture[2]["actor_type"] = "arbiter"
        self.fixture[2]["unexpected"] = "ignored data"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            mod.load_events(self.event_file(self.fixture).read_bytes())


if __name__ == "__main__":
    unittest.main()
