"""Execute the bootstrap's embedded Python without host or cloud operations."""

import base64
import builtins
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


TEMPLATE = Path(__file__).resolve().parents[1] / "user_data.sh.tftpl"
DISABLED_CRON = "# ALIGN smoke tests are disabled.\n"
SERVICE_COMMAND = ["/usr/bin/python3", "/opt/align/run_container.py", "service"]
CRON_COMMAND = ["systemctl", "enable", "--now", "crond"]
SMOKE_CRON = (
    "SHELL=/bin/bash\n"
    "CRON_TZ=America/Toronto\n"
    "0 3 * * * root /usr/bin/python3 /opt/align/run_container.py smoke\n"
)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="align-bootstrap-test-")
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        # Match the base OS directories that exist on Amazon Linux before boot.
        (self.root / "opt").mkdir()
        self.cron_file = self.root / "etc/cron.d/align-smoke"
        self.cron_file.parent.mkdir(parents=True)
        # Model an already-running cron daemon with an existing active job.
        self.cron_file.write_text(SMOKE_CRON)
        timezone = self.root / "usr/share/zoneinfo/America/Toronto"
        timezone.parent.mkdir(parents=True)
        timezone.write_text("fixture timezone")
        self.commands = []
        self.asset_writes = []
        self.fail_service = False

    def mapped_path(self, value):
        path = Path(value)
        allowed = (
            Path("/etc/align"), Path("/opt/align"), Path("/etc/cron.d"),
            Path("/usr/share/zoneinfo"),
        )
        if not any(path == prefix or prefix in path.parents for prefix in allowed):
            raise AssertionError(f"Unexpected bootstrap filesystem access: {value}")
        return self.root / path.relative_to("/")

    def chmod(self, filename, mode):
        path = self.mapped_path(filename)
        os.chmod(path, mode)
        if filename in ("/etc/align/config.json", "/opt/align/run_container.py"):
            self.asset_writes.append((filename, self.cron_file.read_text()))

    def run_command(self, arguments, *, check):
        self.assertTrue(check)
        self.assertIn(arguments, (SERVICE_COMMAND, CRON_COMMAND))
        self.commands.append((list(arguments), self.cron_file.read_text()))
        if arguments == SERVICE_COMMAND:
            self.assertEqual(
                json.loads((self.root / "etc/align/config.json").read_text()),
                self.bundle["config"],
            )
            self.assertEqual(
                (self.root / "opt/align/run_container.py").read_text(),
                self.bundle["runner"],
            )
            if self.fail_service:
                raise subprocess.CalledProcessError(1, arguments)
        return SimpleNamespace(returncode=0)

    def execute_bootstrap(self, *, service, smoke, timezone="America/Toronto"):
        self.bundle = {
            "config": {"service_enabled": service, "smoke_enabled": smoke},
            "runner": "# Fixture runner; subprocess adapter does not execute it.\n",
            "cron": SMOKE_CRON if smoke else DISABLED_CRON,
            "timezone": timezone,
        }
        encoded = base64.b64encode(gzip.compress(json.dumps(self.bundle).encode())).decode()
        template = TEMPLATE.read_text()
        source = template.split("/usr/bin/python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        self.assertEqual(source.count("${runtime_bundle}"), 1)
        source = source.replace("${runtime_bundle}", encoded)
        real_import = builtins.__import__
        adapters = {
            "pathlib": SimpleNamespace(Path=self.mapped_path),
            "os": SimpleNamespace(chmod=self.chmod),
            "subprocess": SimpleNamespace(run=self.run_command),
        }

        def adapted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in adapters:
                return adapters[name]
            return real_import(name, globals, locals, fromlist, level)

        namespace = {"__builtins__": dict(vars(builtins), __import__=adapted_import)}
        exec(compile(source, str(TEMPLATE), "exec"), namespace)

    def assert_assets_written_with_cron_disabled(self):
        self.assertEqual(self.asset_writes, [
            ("/etc/align/config.json", DISABLED_CRON),
            ("/opt/align/run_container.py", DISABLED_CRON),
        ])
        self.assertEqual((self.root / "etc/align/config.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.root / "opt/align/run_container.py").stat().st_mode & 0o777, 0o755)
        self.assertEqual(self.cron_file.stat().st_mode & 0o777, 0o644)
        self.assertFalse((self.cron_file.parent / ".align-smoke.tmp").exists())

    def test_shared_host_starts_service_before_activating_and_enabling_cron(self):
        self.execute_bootstrap(service=True, smoke=True)
        self.assert_assets_written_with_cron_disabled()
        self.assertEqual(self.commands, [
            (SERVICE_COMMAND, DISABLED_CRON),
            (CRON_COMMAND, SMOKE_CRON),
        ])
        self.assertEqual(self.cron_file.read_text(), SMOKE_CRON)

    def test_service_failure_keeps_cron_inactive_and_fails_bootstrap(self):
        self.fail_service = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute_bootstrap(service=True, smoke=True)
        self.assert_assets_written_with_cron_disabled()
        self.assertEqual(self.commands, [(SERVICE_COMMAND, DISABLED_CRON)])
        self.assertEqual(self.cron_file.read_text(), DISABLED_CRON)

    def test_disabled_workloads_do_not_start_service_or_enable_cron(self):
        self.execute_bootstrap(service=False, smoke=False)
        self.assert_assets_written_with_cron_disabled()
        self.assertEqual(self.commands, [])
        self.assertEqual(self.cron_file.read_text(), DISABLED_CRON)

    def test_smoke_only_activates_schedule_without_launching_service(self):
        self.execute_bootstrap(service=False, smoke=True)
        self.assert_assets_written_with_cron_disabled()
        self.assertEqual(self.commands, [(CRON_COMMAND, SMOKE_CRON)])
        self.assertEqual(self.cron_file.read_text(), SMOKE_CRON)

    def test_missing_timezone_keeps_cron_inactive_after_service_starts(self):
        with self.assertRaisesRegex(RuntimeError, "timezone is not installed"):
            self.execute_bootstrap(service=True, smoke=True, timezone="Missing/Timezone")
        self.assert_assets_written_with_cron_disabled()
        self.assertEqual(self.commands, [(SERVICE_COMMAND, DISABLED_CRON)])
        self.assertEqual(self.cron_file.read_text(), DISABLED_CRON)

    def test_service_only_leaves_cron_inactive(self):
        self.execute_bootstrap(service=True, smoke=False)
        self.assert_assets_written_with_cron_disabled()
        self.assertEqual(self.commands, [(SERVICE_COMMAND, DISABLED_CRON)])
        self.assertEqual(self.cron_file.read_text(), DISABLED_CRON)


if __name__ == "__main__":
    unittest.main()
