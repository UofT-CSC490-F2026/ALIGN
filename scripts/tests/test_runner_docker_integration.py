"""Exercise the real Git broker and evaluator against an opt-in local image."""

import json
import os
import subprocess
import unittest
from unittest.mock import patch

from runner.cooperbench_adapter import prepare_environment
from runner.deadline import TrialDeadline
from runner.secure_evaluator import SecureBackend
from runner.shared_git import image_id

IMAGE = os.environ.get("ALIGN_TEST_DOCKER_IMAGE")
BASE = "5c1239b0116b66492cdd0848144ab3c78a04495a"
TASK = {"dataset_source":"cooperbench", "base_commit":BASE,
        "metadata":{"cooperbench_repo_directory":"pallets_click_task", "cooperbench_task_id":2956,
                    "feature_ids":[1,8]},
        "goals":[{"goal_key":"feature1", "owner_agent_id":"agent1"},
                 {"goal_key":"feature8", "owner_agent_id":"agent2"}]}


def inspect(name):
    return json.loads(subprocess.check_output(["docker", "inspect", name], timeout=20))[0]


@unittest.skipUnless(IMAGE, "Set ALIGN_TEST_DOCKER_IMAGE to the pinned Click fixture")
class DockerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.deadline = TrialDeadline(120)
        self.image = image_id(IMAGE, self.deadline)

    def assert_isolated(self, name):
        container = inspect(name)
        config = container["HostConfig"]
        self.assertEqual(container["Image"], self.image)
        self.assertEqual(container["Mounts"], [])
        self.assertEqual(config["NetworkMode"], "none")
        self.assertEqual(config["Memory"], 4 * 1024**3)
        self.assertEqual(config["NanoCpus"], 2_000_000_000)
        self.assertEqual(config["PidsLimit"], 256)
        self.assertIn("ALL", config["CapDrop"])
        self.assertIn("no-new-privileges", config["SecurityOpt"])

    def assert_removed(self, *names):
        for name in names:
            self.assertNotEqual(subprocess.run(["docker", "inspect", name], capture_output=True,
                                               timeout=20).returncode, 0)

    def start_broker(self):
        env = prepare_environment(TASK, self.image, concurrent=True, deadline=self.deadline)
        self.assertEqual(env.image_id,self.image)
        self.addCleanup(self.assert_removed, *env.containers.values(), env._broker)
        self.addCleanup(env.close)
        return env

    def test_publish_fetch_and_merge_actual_commit_ancestry(self):
        env = self.start_broker()
        for name in (*env.containers.values(), env._broker):
            self.assert_isolated(name)
        self.assertEqual(env.state_for("agent1")["branch"], "align-agent1")
        self.assertEqual(env.state_for("agent2")["branch"], "align-agent2")

        def run(actor, tool, refs=None, **args):
            if refs is None:
                refs = env.shared_state()["published"]
            return env.execute(actor, {"tool": tool, **args}, expected_refs=refs)

        initial = env.shared_state()["published"]
        self.assertEqual(run("agent1", "git_publish")["exit_code"], 0)
        self.assertEqual(run("agent2", "git_fetch", peer="agent1")["exit_code"], 0)
        with self.assertRaisesRegex(ValueError, "Invalid Git target"):
            run("agent2", "git_fetch", peer="agent2")
        with self.assertRaisesRegex(ValueError, "Unsupported environment tool"):
            run("agent1", "git_force_push")
        edit = "echo first > align-first.txt && git add align-first.txt && git commit -m first"
        self.assertEqual(run("agent1", "bash", command=edit)["exit_code"], 0)
        self.assertEqual(run("agent1", "git_publish")["exit_code"], 0)
        published = env.shared_state()["published"]
        self.assertNotEqual(published["agent1"], BASE)
        self.assertEqual(published["agent2"], BASE)
        self.assertTrue(run("agent2", "git_fetch", refs=initial, peer="agent1")["stale"])
        with patch("runner.shared_git.MAX_BUNDLE", 32):
            with self.assertRaisesRegex(ValueError, "bundle limit"):
                run("agent2", "git_fetch", peer="agent1")
        self.assertEqual(run("agent2", "git_fetch", peer="agent1")["exit_code"], 0)
        integrate = "git merge --ff-only refs/remotes/align/agent1 && echo second > align-peer.txt"
        self.assertEqual(run("agent2", "bash", command=integrate)["exit_code"], 0)
        self.assertEqual(run("agent1", "bash", command=f"git checkout --detach {BASE}")["exit_code"], 0)
        self.assertIn("Non-fast-forward", run("agent1", "git_publish")["output"])
        self.assertEqual(env.shared_state()["published"], published)
        self.assertEqual(run("agent1", "bash", command="git checkout align-agent1")["exit_code"], 0)
        merged = env.patches()
        self.assertEqual(merged["_merge_status"], "clean")
        self.assertEqual(merged["agent1"], merged["agent2"])
        for filename in ("align-first.txt", "align-peer.txt"):
            self.assertEqual(merged["agent1"].count(f"diff --git a/{filename} "), 1)

    def test_conflicting_submissions_do_not_emit_an_evaluable_patch(self):
        env = self.start_broker()
        for actor in ("agent1", "agent2"):
            outcome = env.execute(actor, {"tool": "bash", "command": f"echo {actor} > align-conflict.txt"})
            self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(env.patches(), {"_merge_status": "conflicts", "agent1": "", "agent2": ""})

    def test_evaluator_streams_patch_and_removes_timed_out_container(self):
        backend = SecureBackend(self.image, BASE, self.deadline)
        self.addCleanup(backend.close)
        sandbox = backend.create_sandbox(self.image, timeout=15)
        self.assert_isolated(sandbox.name)
        sandbox.write_patch("input.patch", "test fixture\n")
        result = sandbox.exec("cat", "/patches/input.patch")
        self.assertEqual(result.stdout_read(), "test fixture\n")
        self.assertEqual(result.stderr_read(), "")
        with self.assertRaisesRegex(ValueError, "Invalid patch filename"):
            sandbox.write_patch("../input.patch", "")
        sandbox.timeout = 1
        with self.assertRaisesRegex(TimeoutError, "Evaluator Docker command timed out"):
            sandbox.exec("sleep", "60")
        self.assert_removed(sandbox.name)
        with self.assertRaisesRegex(RuntimeError, "sandbox closed"):
            sandbox.write_patch("input.patch", "")
        with self.assertRaisesRegex(RuntimeError, "sandbox closed"):
            sandbox.exec("pwd")
