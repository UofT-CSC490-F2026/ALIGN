"""Opt-in integration tests against an empty, explicitly selected local engine.

These run the production helpers and main() with real Docker and a fixture image.
They do not invoke ECR, EC2 bootstrap, cron, or the application's entrypoint.
"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid


MODULE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("align_docker_runtime", MODULE / "scripts/run_container.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)

# Redirect only host paths and service image preparation. Docker commands, signals,
# flock, container execution, and cleanup remain real. Linux MemAvailable is a
# fixture because this client-side runner also executes on macOS.
MAIN_WRAPPER = """
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('align_main_integration', sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
root = Path(sys.argv[2])
runner.LOG_DIRECTORY = root / 'logs'
runner.LOCK_PATH = root / 'locks' / 'smoke.lock'
runner.SCRATCH_DIRECTORY = root / 'scratch'
runner.MEMINFO_PATH = root / 'meminfo'
runner.DOCKER_DATA_DIRECTORY = root
runner.prepare_service_image = lambda config, scratch: ['docker']
def unexpected_preparation(*args):
    raise RuntimeError('Shared smoke attempted image preparation')
runner.prepare_image = unexpected_preparation
sys.argv = [sys.argv[1], sys.argv[3], '--config', sys.argv[4]]
raise SystemExit(runner.main())
"""


@unittest.skipUnless(
    os.environ.get("ALIGN_RUN_DOCKER_INTEGRATION") == "1",
    "opt in with ALIGN_RUN_DOCKER_INTEGRATION=1 and a dedicated Docker context",
)
class DockerIntegrationTests(unittest.TestCase):
    @classmethod
    def command(cls, *arguments, check=True, timeout=30):
        result = subprocess.run(
            ["docker", *arguments], text=True, capture_output=True, timeout=timeout,
        )
        if check and result.returncode:
            raise RuntimeError(f"Docker command failed: {result.stderr.strip()[:1000]}")
        return result

    @classmethod
    def inspect(cls, name):
        return json.loads(cls.command("inspect", name).stdout)[0]

    @classmethod
    def setUpClass(cls):
        if shutil.which("docker") is None:
            raise RuntimeError("Install Docker CLI and start a dedicated local engine first")
        context = os.environ.get("ALIGN_DOCKER_TEST_CONTEXT")
        if not context:
            raise RuntimeError("ALIGN_DOCKER_TEST_CONTEXT must explicitly name a dedicated engine")
        contexts = json.loads(cls.command("context", "inspect", context).stdout)
        endpoint = contexts[0]["Endpoints"]["docker"]["Host"]
        if not endpoint.startswith("unix://"):
            raise RuntimeError("Integration tests require a local unix:// Docker socket")

        cls.workspace = tempfile.TemporaryDirectory(prefix="align-docker-test-")
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.scratch = Path(cls.workspace.name)
        docker_config = cls.scratch / "docker"
        docker_config.mkdir()
        # Homebrew installs Buildx outside Docker's default plugin search paths.
        plugins = [str(path) for path in (
            Path("/opt/homebrew/lib/docker/cli-plugins"),
            Path("/usr/local/lib/docker/cli-plugins"),
        ) if path.is_dir()]
        (docker_config / "config.json").write_text(json.dumps({"cliPluginsExtraDirs": plugins}))
        keys = (
            "DOCKER_CONFIG", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH", "DOCKER_API_VERSION", "DOCKER_BUILDKIT",
            "BUILDX_BUILDER", "BUILDX_CONFIG", "BUILDKIT_HOST",
        )
        cls.previous_environment = {key: os.environ.get(key) for key in keys}
        cls.addClassCleanup(cls.restore_environment)
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(DOCKER_HOST=endpoint, DOCKER_CONFIG=str(docker_config))
        if cls.command("container", "ls", "--all", "--quiet").stdout.strip():
            raise RuntimeError("Selected engine contains containers; use a fresh dedicated engine")

        cls.token = uuid.uuid4().hex
        cls.label = f"align.integration_run={cls.token}"
        cls.image = None
        cls.addClassCleanup(cls.remove_owned_resources)
        cls.image_tag = f"align-runtime-fixture:{cls.token}"
        cls.command(
            "buildx", "build", "--load", "--platform", "linux/amd64",
            "--provenance=false", "--pull", "--label", cls.label,
            "--tag", cls.image_tag, str(MODULE / "tests/docker"), timeout=600,
        )
        cls.image = cls.inspect(cls.image_tag)["Id"]
        image = cls.inspect(cls.image)
        if image["Os"] != "linux" or image["Architecture"] != "amd64":
            raise RuntimeError("Fixture must match the EC2 linux/amd64 platform")
        if image["Config"]["Labels"].get("align.integration_run") != cls.token:
            raise RuntimeError("Fixture ownership label missing")
        engine = json.loads(cls.command("version", "--format", "{{json .Server}}").stdout)
        print(f"\nReal Docker engine {engine['Version']}; fixture linux/amd64; context {context}", flush=True)
        cls.config = {
            "image": cls.image,
            "service_enabled": True,
            "smoke_enabled": True,
            "environment": {"ALIGN_ENVIRONMENT": "dev", "ALIGN_SERVICE_ONLY": "fixture"},
            "smoke_environment": {"ALIGN_ENVIRONMENT": "dev", "ALIGN_SMOKE_OFFLINE": "true"},
            "service_command": ["service"],
            "service_port": None,
            "service_memory_mb": 128,
            "smoke_command": ["smoke-pass"],
            "smoke_cpus": 0.25,
            "smoke_memory_mb": 64,
            "smoke_timeout_seconds": 15,
            "smoke_host_memory_reserve_mb": 64,
            "smoke_host_disk_reserve_mb": 64,
            "smoke_tmpfs_mb": 16,
            "smoke_pids_limit": 32,
        }
        cls.main_root = cls.scratch / "main"
        for name in ("logs", "locks", "scratch"):
            (cls.main_root / name).mkdir(parents=True)
        (cls.main_root / "meminfo").write_text("MemAvailable: 1048576 kB\n")
        cls.docker = ["docker"]
        startup = cls.run_main("service")
        if startup.returncode:
            raise RuntimeError("Fixture main(service) failed: " + startup.stderr)
        cls.service_id = cls.inspect("align-arbiter")["Id"]

    @classmethod
    def main_arguments(cls, mode, config=None):
        path = cls.main_root / f"config-{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(config if config is not None else cls.config))
        return [sys.executable, "-B", "-c", MAIN_WRAPPER, str(MODULE / "scripts/run_container.py"), str(cls.main_root), mode, str(path)]

    @classmethod
    def run_main(cls, mode, config=None):
        return subprocess.run(cls.main_arguments(mode, config), capture_output=True, text=True, timeout=30)

    @classmethod
    def restore_environment(cls):
        for key, value in cls.previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @classmethod
    def remove_owned_resources(cls):
        if cls.image is None:
            probe = cls.command("image", "inspect", cls.image_tag, check=False)
            if probe.returncode == 0:
                image = json.loads(probe.stdout)[0]
                if image["Config"]["Labels"].get("align.integration_run") != cls.token:
                    raise RuntimeError("Refusing to clean up a fixture tag owned by another run")
                cls.image = image["Id"]
        containers = cls.command("container", "ls", "--all", "--quiet", "--filter", f"label={cls.label}").stdout.split()
        for container in containers:
            details = cls.inspect(container)
            if details["Config"]["Labels"].get("align.integration_run") != cls.token:
                raise RuntimeError("Refusing to remove a container without this test's ownership label")
            if cls.image is None or details["Image"] != cls.image:
                raise RuntimeError("Refusing to remove a container with an unexpected image")
            cls.command("rm", "--force", details["Id"])
        if cls.image is not None:
            image = cls.inspect(cls.image)
            if image["Config"]["Labels"].get("align.integration_run") != cls.token:
                raise RuntimeError("Refusing to remove an image without this test's ownership label")
            cls.command("image", "rm", cls.image)

    def service_state(self):
        details = self.inspect(self.service_id)
        self.assertTrue(details["State"]["Running"])
        return details["Id"], details["State"]["StartedAt"]

    def smoke_inputs(self, command):
        config = dict(self.config, smoke_command=[command])
        run_id = uuid.uuid4().hex
        directory = self.scratch / run_id
        directory.mkdir()
        environment = runtime.write_environment(config, directory, run_id)
        self.assertEqual(environment.stat().st_mode & 0o777, 0o600)
        return config, run_id, environment

    def assert_removed(self, run_id):
        remaining = self.command(
            "container", "ls", "--all", "--quiet", "--filter", f"name=^/align-smoke-{run_id}$",
        ).stdout.strip()
        self.assertEqual(remaining, "")

    def wait_until_running(self, name, future):
        deadline = time.monotonic() + 10
        while True:
            probe = self.command("inspect", name, check=False, timeout=5)
            if probe.returncode == 0 and json.loads(probe.stdout)[0]["State"]["Running"]:
                return
            if future.done():
                self.fail(f"Smoke exited before inspection: {future.result()}")
            if time.monotonic() >= deadline:
                self.fail("Smoke container did not start before the inspection deadline")
            time.sleep(0.1)

    def test_service_start_is_idempotent_and_restarts_stopped_container(self):
        before = self.service_state()
        envfile = runtime.write_environment(self.config, self.scratch)
        self.assertEqual(runtime.start_service(self.config, self.docker, envfile), 0)
        self.assertEqual(self.service_state(), before)
        self.command("stop", "--time", "2", self.service_id)
        self.assertEqual(runtime.start_service(self.config, self.docker, envfile), 0)
        self.assertEqual(self.service_state()[0], before[0])
        details = self.inspect(self.service_id)
        self.assertEqual(details["HostConfig"]["RestartPolicy"]["Name"], "unless-stopped")
        self.assertEqual(details["HostConfig"]["LogConfig"]["Type"], "local")
        self.assertEqual(details["Config"]["User"], "65534:65534")

    def test_smoke_success_returns_zero_and_removes_only_its_container(self):
        before = self.service_state()
        runtime.require_clean_smoke_host(time.monotonic() + 10)
        config, run_id, environment = self.smoke_inputs("smoke-pass")
        self.assertEqual(runtime.run_smoke(config, self.docker, environment, run_id, time.monotonic() + 15), 0)
        self.assert_removed(run_id)
        self.assertEqual(self.service_state(), before)

    def test_smoke_failure_preserves_exit_code_and_removes_its_container(self):
        before = self.service_state()
        config, run_id, environment = self.smoke_inputs("smoke-fail")
        self.assertEqual(runtime.run_smoke(config, self.docker, environment, run_id, time.monotonic() + 15), 7)
        self.assert_removed(run_id)
        self.assertEqual(self.service_state(), before)

    def test_smoke_timeout_removes_its_container_and_preserves_service(self):
        before = self.service_state()
        config, run_id, environment = self.smoke_inputs("hold")
        config["smoke_timeout_seconds"] = 3
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runtime.run_smoke, config, self.docker, environment, run_id, time.monotonic() + 15)
            self.wait_until_running(f"align-smoke-{run_id}", future)
            with self.assertRaises(subprocess.TimeoutExpired):
                future.result(timeout=20)
        self.assert_removed(run_id)
        self.assertEqual(self.service_state(), before)

    def test_smoke_resource_limits_and_runtime_environment(self):
        before = self.service_state()
        config, run_id, environment = self.smoke_inputs("hold")
        name = f"align-smoke-{run_id}"
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runtime.run_smoke, config, self.docker, environment, run_id, time.monotonic() + 15)
            self.wait_until_running(name, future)
            details = self.inspect(name)
            host = details["HostConfig"]
            self.assertEqual(host["NanoCpus"], 250000000)
            self.assertEqual(host["Memory"], 64 * 1024 * 1024)
            self.assertEqual(host["MemorySwap"], host["Memory"])
            self.assertEqual(host["LogConfig"], {"Type": "local", "Config": {"max-size": "10m", "max-file": "3"}})
            self.assertFalse(host["Privileged"])
            self.assertTrue(host["ReadonlyRootfs"])
            self.assertEqual(host["NetworkMode"], "none")
            self.assertEqual(host["PidsLimit"], 32)
            self.assertEqual(host["CpuShares"], 128)
            self.assertIn("ALL", host["CapDrop"])
            self.assertIn("no-new-privileges=true", host["SecurityOpt"])
            self.assertEqual(set(host["Tmpfs"]), {"/tmp"})
            self.assertIn("size=16m", host["Tmpfs"]["/tmp"])
            self.assertTrue(all(mount["Type"] == "tmpfs" and mount["Destination"] == "/tmp" for mount in details["Mounts"]))
            self.assertFalse(host.get("PortBindings"))
            self.assertEqual(details["Config"]["Labels"]["align.run_id"], run_id)
            self.assertIn(f"ALIGN_OUTPUT_PREFIX=smoke/{run_id}/", details["Config"]["Env"])
            self.assertIn("ALIGN_SMOKE_OFFLINE=true", details["Config"]["Env"])
            self.assertFalse(any(value.startswith("ALIGN_SERVICE_ONLY=") for value in details["Config"]["Env"]))
            write_probe = self.command("exec", "--user", "0", details["Id"], "touch", "/align-write-probe", check=False)
            self.assertNotEqual(write_probe.returncode, 0)
            self.assertIn("read-only", write_probe.stderr.lower())
            self.command("exec", details["Id"], "touch", "/tmp/align-release")
            self.assertEqual(future.result(timeout=20), 0)
        self.assert_removed(run_id)
        self.assertEqual(self.service_state(), before)

    def test_leftover_running_and_stopped_smoke_containers_block_new_runs(self):
        before = self.service_state()
        config, run_id, environment = self.smoke_inputs("hold")
        container = self.command(
            "create", "--label", "align.workload=smoke", "--label", f"align.run_id={run_id}",
            "--env-file", str(environment), config["image"], "hold",
        ).stdout.strip()
        try:
            self.command("start", container)
            for running in (True, False):
                if not running:
                    self.command("stop", "--time", "2", container)
                with self.assertRaisesRegex(RuntimeError, container[:12]):
                    runtime.require_clean_smoke_host(time.monotonic() + 10)
                self.assertEqual(self.run_main("smoke").returncode, 1)
                self.assertEqual(self.inspect(container)["State"]["Running"], running)
                self.assertEqual(self.service_state(), before)
        finally:
            details = self.inspect(container)
            self.assertEqual(details["Image"], self.image)
            self.assertEqual(details["Config"]["Labels"]["align.integration_run"], self.token)
            self.command("rm", "--force", container)
        runtime.require_clean_smoke_host(time.monotonic() + 10)

    def test_main_service_reuses_existing_container(self):
        before = self.service_state()
        self.assertEqual(self.run_main("service").returncode, 0)
        self.assertEqual(self.service_state(), before)

    def test_main_smoke_exit_codes_and_timeout_preserve_service(self):
        before = self.service_state()
        for command, status in (("smoke-pass", 0), ("smoke-fail", 7), ("hold", 124)):
            with self.subTest(command=command):
                config = dict(self.config, smoke_command=[command], smoke_timeout_seconds=3)
                self.assertEqual(self.run_main("smoke", config).returncode, status)
                runtime.require_clean_smoke_host(time.monotonic() + 10)
                self.assertEqual(self.service_state(), before)

    def test_main_concurrent_smoke_process_skips_under_real_lock(self):
        before = self.service_state()
        config = dict(self.config, smoke_command=["hold"])
        first = subprocess.Popen(self.main_arguments("smoke", config), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 10
            while True:
                ids = self.command("container", "ls", "--quiet", "--filter", "label=align.workload=smoke").stdout.split()
                if ids:
                    break
                if first.poll() is not None:
                    self.fail(f"First smoke exited before starting: {first.communicate()}")
                if time.monotonic() >= deadline:
                    self.fail("First smoke process did not start its container")
                time.sleep(0.1)
            self.assertEqual(len(ids), 1)
            self.assertEqual(self.run_main("smoke").returncode, 0)
            self.assertIsNone(first.poll())
            remaining = self.command("container", "ls", "--quiet", "--filter", "label=align.workload=smoke").stdout.split()
            self.assertEqual(remaining, ids)
            self.assertIn("Skipped smoke job", (self.main_root / "logs/align-smoke.log").read_text())
            self.command("exec", ids[0], "touch", "/tmp/align-release")
            first.communicate(timeout=20)
            self.assertEqual(first.returncode, 0)
        finally:
            if first.poll() is None:
                first.terminate()
                first.communicate(timeout=30)
        runtime.require_clean_smoke_host(time.monotonic() + 10)
        self.assertEqual(self.service_state(), before)

    def test_main_stopped_service_blocks_smoke_without_restarting_it(self):
        self.command("stop", "--time", "2", self.service_id)
        try:
            self.assertEqual(self.run_main("smoke").returncode, 1)
            self.assertFalse(self.inspect(self.service_id)["State"]["Running"])
            runtime.require_clean_smoke_host(time.monotonic() + 10)
        finally:
            self.command("start", self.service_id)

    def test_main_termination_cleans_up_smoke_and_preserves_service(self):
        before = self.service_state()
        child = subprocess.Popen(
            self.main_arguments("smoke", dict(self.config, smoke_command=["hold"])),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                ids = self.command("container", "ls", "--quiet", "--filter", "label=align.workload=smoke").stdout.split()
                if ids:
                    break
                if child.poll() is not None:
                    self.fail(f"Smoke process exited before termination test: {child.communicate()}")
                if time.monotonic() >= deadline:
                    self.fail("Smoke container did not start before termination test")
                time.sleep(0.1)
            child.terminate()
            # Cancellation must clean up before the 15-second smoke deadline.
            child.communicate(timeout=10)
            self.assertEqual(child.returncode, 1)
        finally:
            if child.poll() is None:
                child.terminate()
                child.communicate(timeout=30)
        runtime.require_clean_smoke_host(time.monotonic() + 10)
        self.assertEqual(self.service_state(), before)

    def test_main_missing_service_blocks_smoke_without_recreating_it(self):
        before = self.service_state()
        self.command("rename", self.service_id, f"align-fixture-{self.token}")
        try:
            self.assertEqual(self.run_main("smoke").returncode, 1)
            self.assertEqual(self.command("container", "ls", "--all", "--quiet", "--filter", "name=^/align-arbiter$").stdout.strip(), "")
            runtime.require_clean_smoke_host(time.monotonic() + 10)
            self.assertEqual(self.service_state(), before)
        finally:
            self.command("rename", self.service_id, "align-arbiter")

    def test_main_wrong_service_image_blocks_smoke(self):
        before = self.service_state()
        config = dict(self.config, image="sha256:" + "0" * 64)
        self.assertEqual(self.run_main("smoke", config).returncode, 1)
        runtime.require_clean_smoke_host(time.monotonic() + 10)
        self.assertEqual(self.service_state(), before)

    def test_main_insufficient_headroom_blocks_smoke(self):
        before = self.service_state()
        for key in ("smoke_host_memory_reserve_mb", "smoke_host_disk_reserve_mb"):
            with self.subTest(key=key):
                config = dict(self.config, **{key: 10 ** 12})
                self.assertEqual(self.run_main("smoke", config).returncode, 1)
                runtime.require_clean_smoke_host(time.monotonic() + 10)
                self.assertEqual(self.service_state(), before)


if __name__ == "__main__":
    unittest.main(failfast=True)
