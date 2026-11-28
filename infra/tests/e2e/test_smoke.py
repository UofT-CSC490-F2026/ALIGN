import io
import json
import runpy
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import smoke


class SmokeTests(unittest.TestCase):
    identifier = "12345678-1234-4234-8234-123456789abc"

    def test_sample_is_deterministic_and_scoped(self):
        key, body = smoke.sample(self.identifier)
        self.assertEqual(key, f"smoke/e2e/{self.identifier}/sample.json")
        self.assertEqual((key, body), smoke.sample(self.identifier))

    def test_invalid_ids(self):
        for value in ("../prod", "", self.identifier.upper()):
            with self.assertRaises(ValueError):
                smoke.sample(value)

    def test_wrong_account_rejected(self):
        session = Mock()
        session.client.return_value.get_caller_identity.return_value = {"Account": "wrong"}
        with self.assertRaises(ValueError):
            smoke.check_identity(session, False)

    def test_host_role_required(self):
        session = Mock()
        session.client.return_value.get_caller_identity.return_value = {"Account": smoke.ACCOUNT}
        session.get_credentials.return_value.method = "shared-credentials-file"
        with self.assertRaises(ValueError):
            smoke.check_identity(session, True)
        session.get_credentials.return_value.method = "iam-role"
        smoke.check_identity(session, True)

    def test_round_trip_and_expected_owner(self):
        key, body = smoke.sample(self.identifier)
        client = Mock()
        client.get_object.return_value = {"Body": io.BytesIO(body), "VersionId": "v1"}
        record, _ = smoke.read_sample(client, "bucket", key, body)
        self.assertEqual(record["smoke_id"], self.identifier)
        client.get_object.assert_called_once_with(
            Bucket="bucket", Key=key, ExpectedBucketOwner=smoke.ACCOUNT)

    def test_corruption_and_unversioned_rejected(self):
        _, body = smoke.sample(self.identifier)
        for data, version in ((b"bad", "v1"), (body, "null"), (body, None)):
            client = Mock()
            client.get_object.return_value = {"Body": io.BytesIO(data), "VersionId": version}
            with self.assertRaises(ValueError):
                smoke.read_sample(client, "bucket", "key", body)

    def test_database_missing_or_wrong_row_fails(self):
        for rows in ([], [("failed", "uri", False, "goal")]):
            connection = Mock()
            connection.execute.return_value.fetchall.return_value = rows
            with self.assertRaises(ValueError):
                smoke.verify_database(connection, "key", "uri", "goal")

    def test_database_matching_row_passes(self):
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = [("completed", "uri", True, "goal")]
        smoke.verify_database(connection, "key", "uri", "goal")

    def test_exception_details_are_redacted(self):
        with patch.object(smoke, "run", side_effect=RuntimeError("SECRET-PASSWORD")), \
                patch("sys.stdout", new_callable=io.StringIO), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            code = smoke.main(["verify-backup", "--run-id", self.identifier])
        self.assertEqual(code, 1)
        self.assertNotIn("SECRET-PASSWORD", stderr.getvalue())

    def test_load_uses_downloaded_goal_and_parameterized_values(self):
        connection = Mock()
        connection.execute.return_value.fetchone.return_value = (17,)
        with patch.object(smoke, "verify_database") as verify:
            smoke.load_database(connection, "key", "uri", {"shared_goal": "downloaded goal"})
        calls = connection.execute.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].args[1], ("key", "downloaded goal"))
        self.assertEqual(calls[1].args[1], (17, "key", "uri"))
        verify.assert_called_once_with(connection, "key", "uri", "downloaded goal")

    def test_load_propagates_sql_failure_for_transaction_rollback(self):
        connection = Mock()
        connection.execute.side_effect = RuntimeError("database failure")
        with self.assertRaises(RuntimeError):
            smoke.load_database(connection, "key", "uri", {"shared_goal": "goal"})
        self.assertEqual(connection.execute.call_count, 1)


class SmokeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.identifier = SmokeTests.identifier
        self.key, self.body = smoke.sample(self.identifier)
        self.lake = f"align-dev-lake-{smoke.ACCOUNT}"
        self.uri = f"s3://{self.lake}/{self.key}"
        self.goal = json.loads(self.body)["shared_goal"]
        self.task_key = "e2e-smoke:" + self.identifier
        self.secret_arn = f"arn:aws:secretsmanager:{smoke.REGION}:{smoke.ACCOUNT}:secret:fixture"
        self.db_args = ["--db-host", "db.fixture", "--db-name", "align", "--db-secret-arn",
                        self.secret_arn, "--ca-bundle", __file__]
        self.s3 = Mock()
        self.s3.get_object.return_value = {"Body": io.BytesIO(self.body), "VersionId": "v1"}
        self.secret = Mock()
        self.secret.get_secret_value.return_value = {
            "SecretString": json.dumps({"username": "fixture-user", "password": "fixture-value"})}
        self.aws = Mock()
        session = self.aws.Session.return_value
        sts = Mock()
        sts.get_caller_identity.return_value = {"Account": smoke.ACCOUNT}
        session.get_credentials.return_value.method = "iam-role"
        session.client.side_effect = lambda name, **kwargs: {
            "sts": sts, "s3": self.s3, "secretsmanager": self.secret}[name]
        self.db = Mock()
        self.first, self.second = MagicMock(), MagicMock()
        self.db.connect.side_effect = [
            MagicMock(__enter__=Mock(return_value=c)) for c in (self.first, self.second)]
        for connection in (self.first, self.second):
            connection.execute.return_value.fetchall.return_value = [("completed", self.uri, True, self.goal)]
        dependencies = patch.dict(sys.modules, {
            "boto3": self.aws, "botocore.config": Mock(), "psycopg": self.db})
        dependencies.start()
        self.addCleanup(dependencies.stop)

    def invoke(self, mode, extra):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            code = smoke.main([mode, "--run-id", self.identifier, *extra])
        self.messages = [json.loads(line) for line in output.getvalue().splitlines()]
        return code

    def test_run_writes_versioned_sample_then_checks_committed_row(self):
        self.first.execute.return_value.fetchone.return_value = (11,)
        self.assertEqual(self.invoke("run", self.db_args), 0)
        self.s3.put_object.assert_called_once_with(
            Bucket=self.lake, Key=self.key, Body=self.body, ContentType="application/json",
            IfNoneMatch="*", ExpectedBucketOwner=smoke.ACCOUNT)
        self.secret.get_secret_value.assert_called_once_with(SecretId=self.secret_arn)
        settings = self.db.connect.call_args.kwargs
        self.assertEqual(settings["sslmode"], "verify-full")
        self.assertEqual(settings["sslrootcert"], __file__)
        self.assertEqual(self.db.connect.call_count, 2)
        self.assertEqual(self.first.execute.call_count, 4)
        self.assertEqual(self.second.execute.call_args.args[1], (self.task_key,))
        self.assertEqual([m["stage"] for m in self.messages if "stage" in m],
                         ["s3_readback", "database_and_view"])
        self.assertEqual(self.messages[-1], {"status": "passed", "mode": "run"})

    def test_verify_only_reads_lake_and_database(self):
        self.assertEqual(self.invoke("verify", self.db_args), 0)
        self.s3.put_object.assert_not_called()
        self.s3.get_object.assert_called_once_with(Bucket=self.lake, Key=self.key, ExpectedBucketOwner=smoke.ACCOUNT)
        for connection in (self.first, self.second):
            connection.execute.assert_called_once()
            sql, params = connection.execute.call_args.args
            self.assertTrue(sql.lstrip().startswith("SELECT"))
            self.assertEqual(params, (self.task_key,))

    def test_backup_verifies_replica_using_operator_profile(self):
        self.aws.Session.return_value.get_credentials.return_value.method = "shared-credentials-file"
        self.s3.get_object.return_value["ReplicationStatus"] = "REPLICA"
        self.assertEqual(self.invoke("verify-backup", ["--profile", "operator"]), 0)
        self.aws.Session.assert_called_once_with(region_name=smoke.REGION, profile_name="operator")
        self.s3.get_object.assert_called_once_with(
            Bucket=f"align-dev-backup-{smoke.ACCOUNT}", Key=self.key, ExpectedBucketOwner=smoke.ACCOUNT)
        self.s3.put_object.assert_not_called()
        self.db.connect.assert_not_called()

    def test_backup_rejects_object_without_replica_status(self):
        with patch("sys.stderr", new_callable=io.StringIO) as error:
            self.assertEqual(self.invoke("verify-backup", []), 1)
        self.assertEqual(json.loads(error.getvalue())["error_type"], "ValueError")
        self.assertFalse(any(m.get("status") == "passed" and "stage" not in m for m in self.messages))
        self.db.connect.assert_not_called()

    def test_cli_rejects_invalid_settings_before_aws_access(self):
        cases = [
            (["verify-backup", "--run-id", "invalid"], "canonical lowercase UUID"),
            (["verify", "--run-id", self.identifier], "require DB arguments"),
            (["run", "--run-id", self.identifier, *self.db_args, "--profile", "operator"], "forbid --profile"),
            (["run", "--run-id", self.identifier, *self.db_args[:-1], __file__ + ".missing"], "CA bundle must exist"),
            (["run", "--run-id", self.identifier, *self.db_args, "--db-secret-arn", "wrong"], "intended account/region"),
        ]
        for args, message in cases:
            with self.subTest(message=message), patch.object(sys, "argv", [smoke.__file__, *args]), \
                    patch("sys.stderr", new_callable=io.StringIO) as error, self.assertRaises(SystemExit) as exit_code:
                runpy.run_path(smoke.__file__, run_name="__main__")
            self.assertEqual(exit_code.exception.code, 2)
            self.assertIn(message, error.getvalue())
        self.aws.Session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
