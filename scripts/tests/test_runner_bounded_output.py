"""Exercise actual subprocess capture without Docker or paid inference."""
import subprocess
import sys
import unittest

from runner.docker_utils import (
    OutputLimitExceeded, agent_feedback, bounded_run, require_complete,
)


class BoundedOutputTests(unittest.TestCase):
    def test_drains_both_streams_and_preserves_exit_code(self):
        result = bounded_run(
            [sys.executable, "-c",
             "import sys; sys.stdout.buffer.write(b'a'*300000); "
             "sys.stderr.buffer.write(b'b'*300000); sys.exit(7)"],
            timeout=10, limit=1024,
        )
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, b"a"*1024)
        self.assertEqual(result.stderr, b"b"*1024)
        self.assertTrue(result.output_truncated)
        with self.assertRaises(OutputLimitExceeded):
            require_complete(result)

    def test_exact_output_and_stdin_are_preserved(self):
        data = b"abc"*1000
        result = bounded_run(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
            timeout=10, limit=len(data), data=data, check=True,
        )
        self.assertEqual(result.stdout, data)
        self.assertFalse(result.output_truncated)
        self.assertIs(require_complete(result), result)

    def test_checked_exit_code(self):
        with self.assertRaises(subprocess.CalledProcessError) as error:
            bounded_run([sys.executable, "-c", "raise SystemExit(9)"],
                        timeout=10, check=True)
        self.assertEqual(error.exception.returncode, 9)

    def test_timeout(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            bounded_run([sys.executable, "-c", "import time; time.sleep(5)"],
                        timeout=0.1)

    def test_agent_feedback_marks_display_truncation(self):
        result = subprocess.CompletedProcess([], 3, "x"*13000, "")
        feedback = agent_feedback(result)
        self.assertEqual(feedback["exit_code"], 3)
        self.assertEqual(len(feedback["output"]), 12000)
        self.assertTrue(feedback["output_truncated"])
