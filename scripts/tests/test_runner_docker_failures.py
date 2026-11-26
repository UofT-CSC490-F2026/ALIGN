"""Failure injection at the Docker process boundary."""

import subprocess
import unittest
from threading import Lock
from unittest.mock import Mock, patch

from runner.deadline import TrialDeadline
from runner.secure_evaluator import SecureBackend, SecureSandbox
from runner.shared_git import SharedDockerEnvironment, image_id


class DockerFailureTests(unittest.TestCase):
    def test_image_lookup_rejects_invalid_inputs_and_results(self):
        deadline = TrialDeadline()
        for invalid in (None, "", "--help"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Invalid image reference"):
                image_id(invalid, deadline)
        with patch("runner.shared_git.subprocess.run", return_value=Mock(stdout="invalid")):
            with self.assertRaisesRegex(ValueError, "immutable Docker image ID"):
                image_id("image", deadline)
        with patch("runner.shared_git.subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 1)):
            with self.assertRaisesRegex(TimeoutError, "image inspection timed out"):
                image_id("image", deadline)

    def test_broker_translates_timeouts_and_rejects_oversize_before_io(self):
        env = SharedDockerEnvironment.__new__(SharedDockerEnvironment)
        env.deadline, env.timeout = TrialDeadline(), 30
        with patch("runner.shared_git.MAX_BUNDLE", 2), patch("runner.shared_git.subprocess.run") as docker:
            with self.assertRaisesRegex(ValueError, "bundle limit"):
                env._binary("broker", "cat", b"large")
            docker.assert_not_called()
            docker.side_effect = subprocess.TimeoutExpired("docker", 1)
            with self.assertRaisesRegex(TimeoutError, "Git transfer timed out"):
                env._binary("broker", "cat", b"ok")
        with patch("runner.environment.DockerEnvironment._docker", side_effect=subprocess.TimeoutExpired("docker", 1)):
            with self.assertRaisesRegex(TimeoutError, "Docker command timed out"):
                env._docker(["exec", "broker", "pwd"], timeout=1)

    def test_evaluator_setup_failures_remove_allocated_container(self):
        base = "a" * 40
        for replies, message in (
            ([Mock(returncode=0, stdout="different")], "base commit mismatch"),
            ([Mock(returncode=0, stdout=base), Mock(returncode=0, stdout="dirty")], "workspace must be clean"),
            ([Mock(returncode=0, stdout=base), Mock(returncode=0, stdout=""), Mock(returncode=1)], "directory creation failed"),
        ):
            with self.subTest(message=message), patch.object(SecureSandbox, "_run"), \
                 patch.object(SecureSandbox, "exec", side_effect=replies), \
                 patch("runner.secure_evaluator.subprocess.run", return_value=Mock(returncode=0)) as docker:
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    SecureSandbox("image", base, TrialDeadline(), 1)
                self.assertEqual(docker.call_args.args[0][:3], ["docker", "rm", "-f"])

    def test_backend_validates_identity_and_retries_all_cleanup(self):
        backend = SecureBackend("expected", "a" * 40, TrialDeadline())
        with patch("runner.secure_evaluator.image_id", return_value="different"):
            with self.assertRaisesRegex(ValueError, "image IDs differ"):
                backend.create_sandbox("image")
        self.assertEqual(backend.sandboxes, [])
        first, second = Mock(), Mock()
        first.terminate.side_effect = [OSError("daemon unavailable"), None]
        backend.sandboxes = [first, second]
        with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
            backend.close()
        second.terminate.assert_called_once()
        backend.close()
        self.assertEqual(first.terminate.call_count, 2)

    def test_broker_rejects_unclean_setup_and_malformed_bundle_tip(self):
        base="a"*40
        with patch("runner.environment.DockerEnvironment.__init__", return_value=None), \
             patch.object(SharedDockerEnvironment,"_exec"), patch.object(SharedDockerEnvironment,"_docker"), \
             patch.object(SharedDockerEnvironment,"_broker_exec",return_value=Mock(stdout="unexpected")), \
             patch.object(SharedDockerEnvironment,"close") as close:
            with self.assertRaisesRegex(ValueError,"broker base commit mismatch"):
                SharedDockerEnvironment("image",base,deadline=TrialDeadline())
            close.assert_called_once()

        env=SharedDockerEnvironment.__new__(SharedDockerEnvironment)
        env.base_commit=base
        env._broker="broker"
        with patch.object(env,"_exec",return_value=Mock(stdout="b"*40)), \
             patch.object(env,"_export",return_value=b"bundle"), patch.object(env,"_binary"), \
             patch.object(env,"_broker_exec",return_value=Mock(stdout="invalid tip")) as broker:
            with self.assertRaisesRegex(ValueError,"Invalid imported commit"):
                env._import_broker("agent1")
            self.assertNotIn("update-ref",broker.call_args.args[0])

    def test_broker_cleanup_retries_without_losing_failed_container_name(self):
        env=SharedDockerEnvironment.__new__(SharedDockerEnvironment)
        env._close_lock=Lock()
        env._broker="broker"
        failures=[Mock(returncode=1,stderr="daemon unavailable"),
                  Mock(returncode=1,stderr="No such container: broker")]
        with patch("runner.environment.DockerEnvironment._docker",side_effect=failures), \
             patch("runner.environment.DockerEnvironment.close",side_effect=[OSError(),None,None]) as close:
            with self.assertRaisesRegex(RuntimeError,"Could not remove all trial resources"):
                env.close()
            self.assertEqual(env._broker,"broker")
            env.close()
            self.assertIsNone(env._broker)
            env.close()
            self.assertEqual(close.call_count,3)

    def test_actor_validation_and_non_conflict_merge_failure(self):
        env=SharedDockerEnvironment.__new__(SharedDockerEnvironment)
        env.deadline=TrialDeadline()
        env._git_lock=Lock()
        with self.assertRaisesRegex(ValueError,"Unknown actor"):
            env.state_for("other")
        with self.assertRaisesRegex(ValueError,"Unknown actor"):
            env.execute("other",{"tool":"git_publish"})
        calls=[Mock(returncode=0),Mock(returncode=128),Mock(stdout="")]
        with patch.object(env,"_exec"), patch.object(env,"_import_broker"), \
             patch.object(env,"_broker_exec",side_effect=calls), patch.object(env,"_binary") as export:
            with self.assertRaisesRegex(RuntimeError,"final merge failed"):
                env.patches()
            export.assert_not_called()

    def test_evaluator_cleanup_accepts_already_removed_but_reports_daemon_failure(self):
        sandbox = SecureSandbox.__new__(SecureSandbox)
        sandbox.name, sandbox.closed = "evaluator", False
        failures = [Mock(returncode=1, stderr="daemon unavailable"),
                    Mock(returncode=1, stderr="No such container: evaluator")]
        with patch("runner.secure_evaluator.subprocess.run", side_effect=failures) as docker:
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                sandbox.terminate()
            self.assertFalse(sandbox.closed)
            sandbox.terminate()
            self.assertTrue(sandbox.closed)
            sandbox.terminate()
        self.assertEqual(docker.call_count, 2)
