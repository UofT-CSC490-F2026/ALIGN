import contextlib
import io
import tempfile
import unittest
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


if __name__ == '__main__':
    unittest.main()
