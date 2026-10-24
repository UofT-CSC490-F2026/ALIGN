import io
import unittest
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()
