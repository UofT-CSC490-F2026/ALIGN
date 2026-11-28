import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
from pathlib import Path

import fake_log_demo


class FakeLogArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ca = Path(self.temp.name) / 'ca.pem'
        self.ca.write_text('test fixture, not a real certificate')
        self.args = ['--environment', 'prod', '--db-host',
                     'align-prod-postgres.example.us-east-2.rds.amazonaws.com',
                     '--db-secret-arn', 'arn:aws:secretsmanager:us-east-2:710809468662:secret:rds!db-example',
                     '--ca-bundle', str(self.ca), '--apply']

    def reject(self, args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            fake_log_demo.parse_args(args)
        self.assertEqual(error.exception.code, 2)

    def test_explicit_current_outputs_accepted(self):
        args = fake_log_demo.parse_args(self.args)
        self.assertEqual(args.environment, 'prod')
        self.assertEqual(args.ca_bundle, self.ca)

    def test_apply_required(self):
        self.reject(self.args[:-1])

    def test_environment_host_mismatch_rejected(self):
        args = self.args.copy()
        args[1] = 'dev'
        self.reject(args)

    def test_wrong_account_secret_rejected(self):
        self.reject([x.replace('710809468662', '000000000000') for x in self.args])

    def test_missing_ca_rejected(self):
        self.reject([str(self.ca.with_name('missing.pem')) if x == str(self.ca) else x for x in self.args])

    def test_missing_connection_inputs_rejected(self):
        self.reject(['--environment', 'prod', '--apply'])


class FakeLogWorkflowTests(unittest.TestCase):
    account = "710809468662"
    identifier = "12345678-1234-4234-8234-123456789abc"

    def setUp(self):
        self.bucket = f"align-dev-lake-{self.account}"
        self.key = f"demo/task-log/{self.identifier}/task.json"
        self.uri = f"s3://{self.bucket}/{self.key}"
        self.task_key = f"fake-log:dev:{self.identifier}"
        self.s3, self.sts, self.secret = Mock(), Mock(), Mock()
        self.sts.get_caller_identity.return_value = {"Account": self.account}
        self.secret.get_secret_value.return_value = {
            "SecretString": json.dumps({"username": "fixture-user", "password": "fixture-value"})}
        aws = Mock()
        self.session = aws.Session.return_value
        self.session.client.side_effect = {"s3": self.s3, "sts": self.sts, "secretsmanager": self.secret}.__getitem__
        self.session.get_credentials.return_value.method = "iam-role"
        self.s3.get_object.side_effect = lambda **kwargs: {
            "Body": io.BytesIO(self.s3.put_object.call_args.kwargs["Body"])}
        self.write, self.read = Mock(), Mock()
        self.db = Mock()
        self.db.connect.side_effect = [
            MagicMock(__enter__=Mock(return_value=c)) for c in (self.write, self.read)]
        self.write.execute.side_effect = [
            Mock(fetchone=Mock(return_value=(key,))) for key in (11, 22, 33)]
        self.row = (11, 22, 33, "Add an audio volume setting", "completed",
                    "Fake log: audio volume setting saved successfully.", self.uri)
        self.read.execute.return_value.fetchall.return_value = [self.row]
        dependencies = patch.dict(sys.modules, {
            "boto3": aws, "psycopg": self.db, "psycopg.types.json": Mock(Jsonb=dict)})
        dependencies.start()
        self.addCleanup(dependencies.stop)

    def invoke(self):
        argv = [fake_log_demo.__file__, "--environment", "dev", "--db-host",
                "align-dev-postgres.example.us-east-2.rds.amazonaws.com", "--db-secret-arn",
                f"arn:aws:secretsmanager:us-east-2:{self.account}:secret:rds!db-example",
                "--ca-bundle", __file__, "--apply"]
        with patch.object(sys, "argv", argv), patch("uuid.uuid4", return_value=self.identifier), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            try:
                runpy.run_path(fake_log_demo.__file__, run_name="__main__")
            finally:
                self.messages = [json.loads(line) for line in output.getvalue().splitlines()]

    def test_demo_links_uploaded_task_to_committed_run_and_event(self):
        self.invoke()
        put = self.s3.put_object.call_args.kwargs
        self.assertEqual((put["Bucket"], put["Key"], put["IfNoneMatch"], put["ExpectedBucketOwner"]),
                         (self.bucket, self.key, "*", self.account))
        task = json.loads(put["Body"])
        self.assertEqual(task["task_key"], self.task_key)
        self.s3.get_object.assert_called_once_with(Bucket=self.bucket, Key=self.key, ExpectedBucketOwner=self.account)
        task_insert, run_insert, event_insert = self.write.execute.call_args_list
        self.assertEqual(task_insert.args[1], (self.task_key, task["shared_goal"], {"task_uri": self.uri, "synthetic": True}))
        self.assertEqual(run_insert.args[1], (11, self.identifier, self.uri))
        self.assertEqual(event_insert.args[1], (22, {"message": self.row[5], "synthetic": True}, self.uri))
        self.assertEqual(self.read.execute.call_args.args[1], (self.task_key,))
        self.assertEqual(self.db.connect.call_count, 2)
        self.assertEqual(self.messages, [
            {"stage": "s3_verified", "uri": self.uri},
            {"stage": "committed_sql_verified", "task_id": 11, "run_id": 22, "event_id": 33,
             "message": self.row[5], "task_uri": self.uri}])

    def test_wrong_account_stops_before_writes(self):
        self.sts.get_caller_identity.return_value = {"Account": "wrong"}
        with self.assertRaises(SystemExit) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 1)
        self.s3.put_object.assert_not_called()
        self.db.connect.assert_not_called()

    def test_corrupt_readback_stops_before_database_access(self):
        self.s3.get_object.side_effect = None
        self.s3.get_object.return_value = {"Body": io.BytesIO(b"unexpected")}
        with self.assertRaises(SystemExit) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 1)
        self.secret.get_secret_value.assert_not_called()
        self.db.connect.assert_not_called()
        self.assertEqual(self.messages, [{"status": "failed", "error_type": "AssertionError"}])

    def test_missing_committed_row_cannot_report_success(self):
        self.read.execute.return_value.fetchall.return_value = []
        with self.assertRaises(SystemExit) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.messages, [{"stage": "s3_verified", "uri": self.uri},
                                         {"status": "failed", "error_type": "AssertionError"}])


if __name__ == '__main__':
    unittest.main()
