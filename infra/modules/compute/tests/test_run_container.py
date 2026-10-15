"""Runtime regression tests: no Docker, AWS calls, host files or real sleeps."""

import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch


script = Path(__file__).resolve().parents[1] / "scripts" / "run_container.py"
spec = importlib.util.spec_from_file_location("align_run_container", script)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RuntimeTestCase(unittest.TestCase):
    def replace(self, target, name, **kwargs):
        replacement = patch.object(target, name, **kwargs)
        result = replacement.start()
        self.addCleanup(replacement.stop)
        return result

    def setUp(self):
        self.replace(
            runner.subprocess, "run",
            side_effect=AssertionError("A runtime test attempted a real command"),
        )
        self.replace(runner, "LOG")


class ServiceImageTests(RuntimeTestCase):
    def setUp(self):
        super().setUp()
        self.now = 0.0
        self.delays = []
        self.replace(runner.time, "monotonic", side_effect=lambda: self.now)
        self.replace(runner.time, "sleep", side_effect=self.advance)
        self.prepare = self.replace(runner, "prepare_image", return_value=["docker"])
        self.config = {"image": "mock-image"}
        self.scratch = Path("/mock/runtime")

    def advance(self, seconds):
        self.delays.append(seconds)
        self.now += seconds

    def test_success_does_not_retry(self):
        self.assertEqual(runner.prepare_service_image(self.config, self.scratch), ["docker"])
        self.prepare.assert_called_once_with(self.config, self.scratch, 900)
        self.assertEqual(self.delays, [])

    def test_command_errors_and_timeouts_can_recover(self):
        self.prepare.side_effect = [
            RuntimeError("mock authentication failure"),
            subprocess.TimeoutExpired("mock pull", 300),
            ["docker"],
        ]
        self.assertEqual(runner.prepare_service_image(self.config, self.scratch), ["docker"])
        self.assertEqual(self.prepare.call_count, 3)
        self.assertEqual(self.delays, [5, 10])
        self.assertTrue(all(call.args[2] == 900 for call in self.prepare.call_args_list))

    def test_persistent_failure_stops_after_five_attempts(self):
        self.prepare.side_effect = RuntimeError("mock image unavailable")
        with self.assertRaisesRegex(RuntimeError, "image unavailable"):
            runner.prepare_service_image(self.config, self.scratch)
        self.assertEqual(self.prepare.call_count, 5)
        self.assertEqual(self.delays, [5, 10, 20, 40])

    def test_backoff_cannot_extend_shared_deadline(self):
        self.replace(runner, "SERVICE_IMAGE_TIMEOUT_SECONDS", new=12)
        self.prepare.side_effect = RuntimeError("mock failure")
        with self.assertRaises(subprocess.TimeoutExpired):
            runner.prepare_service_image(self.config, self.scratch)
        self.assertEqual(self.now, 12)
        self.assertEqual(self.delays, [5, 7])
        self.assertEqual(self.prepare.call_count, 2)
        self.assertTrue(all(call.args[2] == 12 for call in self.prepare.call_args_list))

    def test_interruption_missing_commands_and_config_errors_are_not_retried(self):
        for error in (InterruptedError("stop"), FileNotFoundError("aws"), KeyError("image")):
            with self.subTest(error=type(error).__name__):
                self.prepare.reset_mock()
                self.prepare.side_effect = error
                with self.assertRaises(type(error)):
                    runner.prepare_service_image(self.config, self.scratch)
                self.prepare.assert_called_once()
                self.assertEqual(self.delays, [])


class SmokeHostTests(RuntimeTestCase):
    def setUp(self):
        super().setUp()
        self.execute = self.replace(runner, "execute", return_value=SimpleNamespace(stdout=""))

    def test_all_labelled_container_states_are_checked(self):
        runner.require_clean_smoke_host(900)
        self.execute.assert_called_once_with([
            "docker", "container", "ls", "--all", "--filter", "label=align.workload=smoke",
            "--format", "{{.ID}}",
        ], deadline=900, limit=30)

    def test_leftovers_fail_without_removing_them(self):
        self.execute.return_value.stdout = "abc123\ndef456\n"
        with self.assertRaisesRegex(RuntimeError, "abc123, def456"):
            runner.require_clean_smoke_host(900)
        self.execute.assert_called_once()

    def test_listing_failure_is_not_treated_as_an_empty_host(self):
        self.execute.side_effect = RuntimeError("mock Docker unavailable")
        with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
            runner.require_clean_smoke_host(900)
        self.execute.assert_called_once()


class RunnerRoutingTests(RuntimeTestCase):
    def setUp(self):
        super().setUp()
        self.config = {
            "service_enabled": False,
            "smoke_enabled": True,
            "image": "mock-image",
            "smoke_command": ["mock-smoke"],
            "environment": {"ALIGN_ENVIRONMENT": "dev"},
            "smoke_timeout_seconds": 900,
        }
        self.replace(runner, "RotatingFileHandler")
        self.replace(runner.logging, "basicConfig")
        self.replace(runner.signal, "signal")
        self.replace(runner.os, "umask")
        self.replace(runner.time, "monotonic", return_value=100)
        self.read_config = self.replace(runner.Path, "read_text")
        self.lock = self.replace(builtins, "open", new=mock_open()).return_value
        self.flock = self.replace(runner.fcntl, "flock")
        self.scratch = self.replace(runner.tempfile, "TemporaryDirectory")
        self.scratch.return_value.__enter__.return_value = "/mock/runtime"
        self.execute = self.replace(runner, "execute", return_value=SimpleNamespace(stdout=""))
        self.smoke_image = self.replace(runner, "prepare_image", return_value=["docker"])
        self.service_image = self.replace(runner, "prepare_service_image", return_value=["docker"])
        self.replace(runner, "write_environment", return_value=Path("/mock/runtime/env"))
        self.service = self.replace(runner, "start_service", return_value=0)
        self.smoke = self.replace(runner, "run_smoke", return_value=0)

    def invoke(self, mode="smoke"):
        self.replace(sys, "argv", new=["run_container.py", mode])
        self.read_config.return_value = json.dumps(self.config)
        return runner.main()

    def check_preflight_order(self, *_args, **_kwargs):
        self.flock.assert_called_once()
        self.smoke_image.assert_not_called()
        self.service_image.assert_not_called()

    def assert_no_workload_started(self):
        self.smoke_image.assert_not_called()
        self.service_image.assert_not_called()
        self.service.assert_not_called()
        self.smoke.assert_not_called()
        self.scratch.assert_not_called()
        self.lock.close.assert_called_once()

    def test_leftover_container_blocks_before_ecr_or_container_creation(self):
        def leftovers(*args, **kwargs):
            self.check_preflight_order()
            return SimpleNamespace(stdout="abc123\n")

        self.execute.side_effect = leftovers
        self.assertEqual(self.invoke(), 1)
        self.assert_no_workload_started()

    def test_listing_error_blocks_before_ecr_or_container_creation(self):
        def unavailable(*args, **kwargs):
            self.check_preflight_order()
            raise RuntimeError("mock Docker unavailable")

        self.execute.side_effect = unavailable
        self.assertEqual(self.invoke(), 1)
        self.assert_no_workload_started()

    def test_clean_host_proceeds_only_after_locked_preflight(self):
        def clean(*args, **kwargs):
            self.check_preflight_order()
            return SimpleNamespace(stdout="")

        def prepare(*args, **kwargs):
            self.execute.assert_called_once()
            return ["docker"]

        self.execute.side_effect = clean
        self.smoke_image.side_effect = prepare
        self.assertEqual(self.invoke(), 0)
        self.smoke_image.assert_called_once_with(self.config, Path("/mock/runtime"), 1000)
        self.smoke.assert_called_once()
        self.service.assert_not_called()
        self.lock.close.assert_called_once()

    def test_busy_lock_skips_without_docker_or_ecr_calls(self):
        self.flock.side_effect = BlockingIOError()
        self.assertEqual(self.invoke(), 0)
        self.execute.assert_not_called()
        self.assert_no_workload_started()

    def test_service_uses_preparation_retries_without_smoke_preflight(self):
        self.config.update(service_enabled=True, smoke_enabled=False)
        self.assertEqual(self.invoke("service"), 0)
        self.service_image.assert_called_once_with(self.config, Path("/mock/runtime"))
        self.service.assert_called_once()
        self.execute.assert_not_called()
        self.flock.assert_not_called()
        self.smoke_image.assert_not_called()
        self.smoke.assert_not_called()

    def test_service_container_launch_is_not_retried(self):
        self.config.update(service_enabled=True, smoke_enabled=False)
        self.service.side_effect = RuntimeError("mock ambiguous launch failure")
        self.assertEqual(self.invoke("service"), 1)
        self.service_image.assert_called_once()
        self.service.assert_called_once()
        self.smoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
