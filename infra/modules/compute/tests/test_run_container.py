"""Runtime regression tests: no Docker, AWS calls, host files or real sleeps."""

import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
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
        for error in (
            runner.RunnerInterrupted("stop"), InterruptedError("stop"),
            FileNotFoundError("aws"), KeyError("image"),
        ):
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


class HostHeadroomTests(RuntimeTestCase):
    def setUp(self):
        super().setUp()
        self.config = {
            "smoke_memory_mb": 64, "smoke_host_memory_reserve_mb": 32,
            "smoke_host_disk_reserve_mb": 128,
        }
        self.memory = self.replace(runner.Path, "read_text", return_value="MemAvailable: 98304 kB\n")
        self.disk = self.replace(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=128 * 1024 * 1024))

    def test_exact_headroom_is_accepted(self):
        runner.require_host_headroom(self.config)
        self.disk.assert_called_once_with(runner.DOCKER_DATA_DIRECTORY)

    def test_low_memory_blocks_before_disk_check(self):
        self.memory.return_value = "MemAvailable: 98303 kB\n"
        with self.assertRaisesRegex(RuntimeError, "memory headroom"):
            runner.require_host_headroom(self.config)
        self.disk.assert_not_called()

    def test_missing_memory_information_fails_closed(self):
        self.memory.return_value = "MemFree: 98304 kB\n"
        with self.assertRaisesRegex(RuntimeError, "Cannot determine"):
            runner.require_host_headroom(self.config)
        self.disk.assert_not_called()

    def test_low_disk_blocks_smoke(self):
        self.disk.return_value.free -= 1
        with self.assertRaisesRegex(RuntimeError, "disk headroom"):
            runner.require_host_headroom(self.config)


class SharedServiceTests(RuntimeTestCase):
    def setUp(self):
        super().setUp()
        self.config = {"image": "mock-image", "service_memory_mb": 128}
        self.details = {
            "Config": {"Image": "mock-image", "Labels": {"align.workload": "service"}},
            "HostConfig": {"Memory": 128 * 1024 * 1024, "MemorySwap": 128 * 1024 * 1024},
            "State": {"Running": True}, "Image": "mock-image-id",
        }
        self.execute = self.replace(runner, "execute", side_effect=self.inspect)

    def inspect(self, arguments, **_kwargs):
        if arguments[1:3] == ["image", "inspect"]:
            return SimpleNamespace(stdout=json.dumps([{"Id": "mock-image-id"}]))
        return SimpleNamespace(stdout=json.dumps([self.details]))

    def test_running_service_uses_matching_cached_image_without_mutations(self):
        runner.require_running_service(self.config, 900)
        self.assertEqual(self.execute.call_count, 2)
        self.assertTrue(all("inspect" in call.args[0] for call in self.execute.call_args_list))

    def test_stopped_wrong_image_foreign_or_wrong_budget_service_is_rejected(self):
        for field, value in (
            ("stopped", False), ("image", "other-image"), ("label", "other-workload"), ("budget", 0),
        ):
            with self.subTest(field=field):
                details = json.loads(json.dumps(self.details))
                if field == "stopped":
                    details["State"]["Running"] = value
                elif field == "image":
                    details["Config"]["Image"] = value
                elif field == "label":
                    details["Config"]["Labels"]["align.workload"] = value
                else:
                    details["HostConfig"]["Memory"] = value
                self.execute.side_effect = None
                self.execute.return_value = SimpleNamespace(stdout=json.dumps([details]))
                self.execute.reset_mock()
                with self.assertRaises(RuntimeError):
                    runner.require_running_service(self.config, 900)
                self.execute.assert_called_once()

    def test_missing_cached_image_fails_without_pulling(self):
        self.execute.side_effect = [
            SimpleNamespace(stdout=json.dumps([self.details])), RuntimeError("missing image"),
        ]
        with self.assertRaisesRegex(RuntimeError, "missing image"):
            runner.require_running_service(self.config, 900)
        self.assertTrue(all("inspect" in call.args[0] for call in self.execute.call_args_list))


class OfflineEnvironmentTests(RuntimeTestCase):
    def test_smoke_does_not_inherit_service_settings(self):
        config = {
            "environment": {"ALIGN_DB_SECRET_ARN": "fixture-reference", "SERVICE_SETTING": "service"},
            "smoke_environment": {"ALIGN_ENVIRONMENT": "dev", "TEST_SETTING": "fixture"},
        }
        with tempfile.TemporaryDirectory(prefix="align-env-test-") as directory:
            path = runner.write_environment(config, Path(directory), "fixture-run")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(dict(line.split("=", 1) for line in path.read_text().splitlines()), {
                "ALIGN_ENVIRONMENT": "dev", "TEST_SETTING": "fixture", "ALIGN_SMOKE_OFFLINE": "true",
                "ALIGN_RUN_ID": "fixture-run", "ALIGN_OUTPUT_PREFIX": "smoke/fixture-run/",
            })


class SmokeImageTests(RuntimeTestCase):
    def test_declared_volume_is_rejected_before_creating_any_container(self):
        execute = self.replace(runner, "execute", return_value=SimpleNamespace(
            stdout=json.dumps([{"Config": {"Volumes": {"/data": {}}}}]),
        ))
        with self.assertRaisesRegex(RuntimeError, "VOLUME"):
            runner.run_smoke({"image": "fixture-image"}, ["docker"], Path("/mock/env"), "fixture-run", 900)
        execute.assert_called_once_with(["docker", "image", "inspect", "fixture-image"], deadline=900, limit=30)


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
        self.headroom = self.replace(runner, "require_host_headroom")
        self.running_service = self.replace(runner, "require_running_service")
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

    def test_shared_smoke_uses_cached_image_after_locked_preflight(self):
        self.config.update(service_enabled=True, service_memory_mb=128)

        def service_guard(*_args):
            self.flock.assert_called_once()
            self.headroom.assert_called_once_with(self.config)
            self.scratch.assert_not_called()

        self.running_service.side_effect = service_guard
        self.assertEqual(self.invoke(), 0)
        self.running_service.assert_called_once_with(self.config, 1000)
        self.smoke_image.assert_not_called()
        self.service_image.assert_not_called()
        self.service.assert_not_called()
        self.assertEqual(self.smoke.call_args.args[1], ["docker"])

    def test_shared_service_still_uses_startup_image_preparation(self):
        self.config.update(service_enabled=True, service_memory_mb=128)
        self.assertEqual(self.invoke("service"), 0)
        self.service_image.assert_called_once()
        self.service.assert_called_once()
        self.headroom.assert_not_called()
        self.running_service.assert_not_called()
        self.flock.assert_not_called()

    def test_headroom_or_service_failure_prevents_preparation_and_smoke(self):
        self.config.update(service_enabled=True, service_memory_mb=128)
        for guard in (self.headroom, self.running_service):
            with self.subTest(guard=guard):
                guard.side_effect = RuntimeError("preflight failure")
                self.assertEqual(self.invoke(), 1)
                self.smoke.assert_not_called()
                self.service.assert_not_called()
                self.smoke_image.assert_not_called()
                self.service_image.assert_not_called()
                self.scratch.assert_not_called()
                guard.side_effect = None

    def test_shared_mode_rejects_missing_service_budget_and_prod(self):
        self.config.update(service_enabled=True)
        self.assertEqual(self.invoke(), 1)
        self.config.update(service_memory_mb=128, environment={"ALIGN_ENVIRONMENT": "prod"})
        self.assertEqual(self.invoke(), 1)
        self.flock.assert_not_called()
        self.execute.assert_not_called()
        self.smoke_image.assert_not_called()
        self.service_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
