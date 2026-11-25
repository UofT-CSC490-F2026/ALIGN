"""Exercise actual subprocess capture without Docker or paid inference."""
import subprocess
import sys
import unittest
from io import BytesIO
from threading import Event, Thread
from unittest.mock import Mock, patch

from runner.docker_utils import (
    OutputLimitExceeded, agent_feedback, bounded_run, require_complete,
)


class BoundedOutputTests(unittest.TestCase):
    def test_rejects_invalid_limits_before_launching_process(self):
        for options in ({"timeout":0},{"timeout":1,"limit":0},{"timeout":1,"limit":True}):
            with self.subTest(options=options), patch("runner.docker_utils.subprocess.Popen") as launch:
                with self.assertRaisesRegex(ValueError,"Positive output limit"):
                    bounded_run([sys.executable],**options)
                launch.assert_not_called()

    def test_pipe_read_error_does_not_produce_a_complete_result(self):
        class BrokenPipe(BytesIO):
            def read(self,*args):
                raise OSError("fixture pipe error")

        process=Mock(returncode=0,stdout=BrokenPipe(),stderr=BytesIO(b"exit message"))
        process.poll.return_value=0
        with patch("runner.docker_utils.subprocess.Popen",return_value=process):
            with self.assertRaisesRegex(RuntimeError,"output capture failed") as error:
                bounded_run([sys.executable],timeout=5)
        self.assertIsInstance(error.exception.__cause__,OSError)

    def test_exit_with_pipe_still_open_respects_capture_deadline(self):
        release=Event()

        class StillOpenPipe(BytesIO):
            def read(self,*args):
                release.wait(5)
                return b""

        process=Mock(returncode=0,stdout=StillOpenPipe(),stderr=BytesIO())
        process.poll.return_value=0
        try:
            with patch("runner.docker_utils.subprocess.Popen",return_value=process):
                with self.assertRaises(subprocess.TimeoutExpired):
                    bounded_run([sys.executable],timeout=0.1)
        finally:
            release.set()
        process.kill.assert_not_called()

    def test_second_reader_start_failure_reaps_child(self):
        started=[]

        class Reader(Thread):
            def start(self):
                if started:
                    raise RuntimeError("cannot start second reader")
                started.append(self)
                super().start()

        process=Mock(stdout=BytesIO(),stderr=BytesIO())
        process.poll.return_value=None
        with patch("runner.docker_utils.subprocess.Popen",return_value=process), \
             patch("runner.docker_utils.Thread",Reader):
            with self.assertRaisesRegex(RuntimeError,"cannot start second reader"):
                bounded_run([sys.executable],timeout=5)
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        self.assertFalse(started[0].is_alive())
        self.assertTrue(process.stderr.closed)

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
