"""Offline tests for run bundle integrity and event contract."""
import io
import runpy
import sys
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError
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

    def cli(self, *args):
        with patch.object(sys, "argv", [mod.__file__, *map(str, args)]), patch("sys.stdout", new_callable=io.StringIO) as output:
            runpy.run_path(mod.__file__, run_name="__main__")
        return json.loads(output.getvalue())

    def test_invalid_uuid_and_observation_timestamp(self):
        self.fixture[0]["observed_at"] = "invalidZ"
        with self.assertRaisesRegex(ValueError, "Invalid timestamp"):
            mod.validate_events(self.fixture)
        with self.assertRaisesRegex(ValueError, "Invalid canonical UUID"):
            mod.canonical_uuid("invalid")

    def test_artifact_cli_round_trip_and_idempotent_upload(self):
        bundle = self.root / "bundle"
        manifest = self.cli("stage", "--events", EXAMPLE, "--output-dir", bundle)
        self.assertEqual(self.cli("verify", "--run-dir", bundle), manifest)
        objects = {}
        account, bucket = "fixture-account", "fixture-lake"

        def read(*, Bucket, Key, ExpectedBucketOwner):
            self.assertEqual((Bucket, ExpectedBucketOwner), (bucket, account))
            if Key not in objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": io.BytesIO(objects[Key])}

        def write(*, Bucket, Key, Body, ContentType, IfNoneMatch, ExpectedBucketOwner):
            self.assertEqual((Bucket, ExpectedBucketOwner, IfNoneMatch), (bucket, account, "*"))
            self.assertNotIn(Key, objects)
            objects[Key] = Body

        s3, sts = Mock(), Mock()
        s3.get_object.side_effect, s3.put_object.side_effect = read, write
        sts.get_caller_identity.return_value = {"Account": account}
        with patch("boto3.Session") as session:
            session.return_value.client.side_effect = {"sts": sts, "s3": s3}.__getitem__
            flags = ["--bucket", bucket, "--expected-account", account]
            upload = ["upload", "--run-dir", bundle, *flags]
            with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                self.cli(*upload)
            session.assert_not_called()
            uri = f"s3://{bucket}/runs/{manifest['run_key']}/"
            self.assertEqual(self.cli(*upload, "--apply"), uri)
            self.assertEqual(self.cli(*upload, "--apply"), uri)
            self.assertEqual(s3.put_object.call_count, 2)
            target = self.root / "download"
            self.assertEqual(self.cli("download", "--run-key", manifest["run_key"], "--output-dir", target, *flags), manifest)
            self.assertEqual((target / "events.jsonl").read_bytes(), EXAMPLE.read_bytes())
            s3.get_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
            with self.assertRaisesRegex(ClientError, "AccessDenied"):
                mod.upload(bundle, bucket, account)
            self.assertEqual(s3.put_object.call_count, 2)

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
