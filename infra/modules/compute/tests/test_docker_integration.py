"""Opt-in integration tests against an empty, explicitly selected local engine.

These run the production container helpers with real Docker and a fixture image.
They do not invoke ECR, EC2 bootstrap, cron, or the application's entrypoint.
"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid


MODULE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("align_docker_runtime", MODULE / "scripts/run_container.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


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
            "environment": {"ALIGN_ENVIRONMENT": "dev"},
            "service_command": ["service"],
            "service_port": None,
            "smoke_command": ["smoke-pass"],
            "smoke_cpus": 0.25,
            "smoke_memory_mb": 64,
            "smoke_timeout_seconds": 15,
        }
        cls.docker = ["docker"]
        runtime.start_service(cls.config, cls.docker, runtime.write_environment(cls.config, cls.scratch))
        cls.service_id = cls.inspect("align-arbiter")["Id"]

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
            self.assertEqual(details["Mounts"], [])
            self.assertFalse(host.get("PortBindings"))
            self.assertEqual(details["Config"]["Labels"]["align.run_id"], run_id)
            self.assertIn(f"ALIGN_OUTPUT_PREFIX=smoke/{run_id}/", details["Config"]["Env"])
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
                self.assertEqual(self.inspect(container)["State"]["Running"], running)
                self.assertEqual(self.service_state(), before)
        finally:
            details = self.inspect(container)
            self.assertEqual(details["Image"], self.image)
            self.assertEqual(details["Config"]["Labels"]["align.integration_run"], self.token)
            self.command("rm", "--force", container)
        runtime.require_clean_smoke_host(time.monotonic() + 10)


if __name__ == "__main__":
    unittest.main(failfast=True)
