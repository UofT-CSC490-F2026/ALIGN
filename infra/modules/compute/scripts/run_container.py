#!/usr/bin/python3
"""Run a published image on EC2. Application code and secret retrieval live in it."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid


LOG = logging.getLogger("align")
SERVICE_IMAGE_ATTEMPTS = 5
SERVICE_IMAGE_TIMEOUT_SECONDS = 900


def interrupted(signum, _frame):
    # Unwind through the smoke cleanup block when the runner is stopped.
    raise InterruptedError(f"Runner received signal {signum}")


def execute(arguments, *, deadline=None, input_text=None, check=True, limit=300):
    remaining = limit if deadline is None else min(limit, deadline - time.monotonic())
    if remaining <= 0:
        raise subprocess.TimeoutExpired(arguments, 0)
    result = subprocess.run(
        arguments, input=input_text, text=True, capture_output=True, timeout=remaining,
    )
    if check and result.returncode:
        # Never log stdout: get-login-password returns a credential on stdout.
        raise RuntimeError(f"{arguments[0]} failed ({result.returncode}): {result.stderr.strip()[:1000]}")
    return result


def prepare_image(config, scratch, deadline):
    docker = ["docker", "--config", str(scratch / "docker")]
    password = execute(
        ["aws", "ecr", "get-login-password", "--region", config["region"]],
        deadline=deadline,
    ).stdout
    execute(
        docker + ["login", "--username", "AWS", "--password-stdin", config["registry"]],
        deadline=deadline, input_text=password,
    )
    execute(docker + ["pull", "--platform", "linux/amd64", config["image"]], deadline=deadline)
    return docker


def prepare_service_image(config, scratch):
    deadline = time.monotonic() + SERVICE_IMAGE_TIMEOUT_SECONDS
    for attempt in range(1, SERVICE_IMAGE_ATTEMPTS + 1):
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired("ECR image preparation", SERVICE_IMAGE_TIMEOUT_SECONDS)
        try:
            return prepare_image(config, scratch, deadline)
        except (RuntimeError, subprocess.TimeoutExpired):
            remaining = deadline - time.monotonic()
            if attempt == SERVICE_IMAGE_ATTEMPTS or remaining <= 0:
                raise
            delay = min(5 * 2 ** (attempt - 1), remaining)
            LOG.warning(
                "Image preparation attempt %s/%s failed; retrying in %.1f seconds",
                attempt, SERVICE_IMAGE_ATTEMPTS, delay,
            )
            time.sleep(delay)


def require_clean_smoke_host(deadline):
    # The process lock disappears if the runner dies; its container may survive.
    # Check all states, and leave previous containers untouched for review.
    existing = execute([
        "docker", "container", "ls", "--all", "--filter", "label=align.workload=smoke",
        "--format", "{{.ID}}",
    ], deadline=deadline, limit=30).stdout.split()
    if existing:
        raise RuntimeError(
            f"Previous smoke containers remain ({', '.join(existing)}); "
            "review containers labelled align.workload=smoke before retrying"
        )


def write_environment(config, scratch, run_id=None):
    environment = dict(config["environment"])
    if run_id is not None:
        environment.update(ALIGN_RUN_ID=run_id, ALIGN_OUTPUT_PREFIX=f"smoke/{run_id}/")
    path = scratch / "runtime.env"
    path.write_text("".join(f"{key}={value}\n" for key, value in environment.items()))
    path.chmod(0o600)
    return path


def start_service(config, docker, environment_file):
    name = "align-arbiter"
    existing = execute([
        "docker", "container", "ls", "--all", "--filter", f"name=^/{name}$", "--format", "{{.ID}}",
    ]).stdout.strip()
    if existing:
        details = json.loads(execute(["docker", "inspect", existing]).stdout)[0]
        labels = details["Config"].get("Labels") or {}
        if labels.get("align.workload") != "service" or details["Config"]["Image"] != config["image"]:
            raise RuntimeError("An existing arbiter container differs; use an explicit deployment to replace it")
        if not details["State"]["Running"]:
            execute(["docker", "start", existing])
        LOG.info("Regular arbiter container is running")
        return 0

    arguments = docker + [
        "run", "--detach", "--name", name, "--restart", "unless-stopped",
        "--label", "align.workload=service", "--env-file", str(environment_file),
        "--log-driver", "local", "--log-opt", "max-size=10m", "--log-opt", "max-file=3",
    ]
    if config["service_port"] is not None:
        port = config["service_port"]
        arguments += ["--publish", f"127.0.0.1:{port}:{port}"]
    execute(arguments + [config["image"]] + config["service_command"])
    LOG.info("Started regular arbiter container")
    return 0


def run_smoke(config, docker, environment_file, run_id, deadline):
    name = f"align-smoke-{run_id}"
    created = False
    try:
        memory = f"{config['smoke_memory_mb']}m"
        arguments = docker + [
            "create", "--name", name, "--label", "align.workload=smoke",
            "--label", f"align.run_id={run_id}",
            "--cpus", str(config["smoke_cpus"]), "--memory", memory, "--memory-swap", memory,
            "--env-file", str(environment_file), "--log-driver", "local",
            "--log-opt", "max-size=10m", "--log-opt", "max-file=3",
            config["image"],
        ]
        execute(arguments + config["smoke_command"], deadline=deadline)
        created = True
        execute(["docker", "start", name], deadline=deadline)
        # Killing the docker-wait client does not stop its container. The finally
        # block below removes only this uniquely named test container on timeout.
        result = execute(["docker", "wait", name], deadline=deadline, limit=config["smoke_timeout_seconds"])
        status = int(result.stdout.strip())
        logs = execute(["docker", "logs", "--tail", "200", name], check=False, limit=30)
        if logs.stdout or logs.stderr:
            LOG.info("Smoke container output:\n%s%s", logs.stdout, logs.stderr)
        LOG.info("Smoke run %s finished with exit status %s", run_id, status)
        return status
    finally:
        # Handle a create timeout too: the daemon might have created the container
        # even if the CLI timed out before reporting success.
        if not created:
            probe = execute(["docker", "inspect", name], check=False, limit=30)
            if probe.returncode == 0:
                labels = json.loads(probe.stdout)[0]["Config"].get("Labels") or {}
                created = labels.get("align.run_id") == run_id
        if created:
            execute(["docker", "rm", "--force", name], limit=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["service", "smoke"])
    parser.add_argument("--config", default="/etc/align/config.json")
    arguments = parser.parse_args()
    os.umask(0o077)
    handler = RotatingFileHandler(
        f"/var/log/align-{arguments.mode}.log", maxBytes=10 * 1024 * 1024, backupCount=3,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[handler])
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    lock_file = None
    try:
        config = json.loads(Path(arguments.config).read_text())
        if config["service_enabled"] and config["smoke_enabled"]:
            raise RuntimeError("Service and smoke workloads must use separate deployments")
        enabled = config["service_enabled"] if arguments.mode == "service" else config["smoke_enabled"]
        if not enabled:
            LOG.info("%s workload is disabled", arguments.mode)
            return 0
        if not config["image"]:
            raise RuntimeError("A published image digest is required")

        deadline = None
        if arguments.mode == "smoke":
            if config["environment"]["ALIGN_ENVIRONMENT"] != "dev" or not config["smoke_command"]:
                raise RuntimeError("Smoke jobs need a dedicated dev deployment and an explicit command")
            lock_file = open("/run/lock/align-smoke.lock", "a")
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                LOG.info("Skipped smoke job because a previous run is active")
                return 0
            deadline = time.monotonic() + config["smoke_timeout_seconds"]
            require_clean_smoke_host(deadline)

        with tempfile.TemporaryDirectory(prefix="align-runtime-", dir="/run") as directory:
            scratch = Path(directory)
            if arguments.mode == "service":
                docker = prepare_service_image(config, scratch)
                return start_service(config, docker, write_environment(config, scratch))
            docker = prepare_image(config, scratch, deadline)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
            return run_smoke(config, docker, write_environment(config, scratch, run_id), run_id, deadline)
    except subprocess.TimeoutExpired:
        LOG.error("%s workload timed out", arguments.mode)
        return 124
    except Exception as error:
        LOG.error("%s workload failed: %s", arguments.mode, error)
        return 1
    finally:
        if lock_file is not None:
            lock_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
