import unittest
import json
from pathlib import Path
from unittest.mock import patch, Mock
from importlib import metadata
from runner.cooperbench_adapter import installed_revision, prepare_environment
from runner.environment import MockEnvironment, ScriptedAgent
from runner.task_loader import agent_view

class EnvironmentTests(unittest.TestCase):
    def test_projection(self):
        t={"task_key":"x","shared_goal":"both","constraints":{"secret":"hidden"},
           "goals":[{"owner_type":"agent","owner_agent_id":f"agent{i}","description":"edit",
                     "success_criteria":{"secret":"hidden"}} for i in (1,2)]}
        self.assertNotIn("hidden",str(agent_view(t)))

    def test_projection_excludes_non_agent_goals_and_requires_both_agents(self):
        fixture={"task_key":"fixture","shared_goal":"goal","constraints":{},"goals":[
            {"owner_type":"shared","description":"evaluator-only"},
            {"owner_type":"agent","owner_agent_id":"agent1","description":"first"},
            {"owner_type":"agent","owner_agent_id":"agent2","description":"second"},
        ]}
        self.assertEqual(agent_view(fixture)["agent_goals"],{"agent1":"first","agent2":"second"})
        fixture["goals"].pop()
        with self.assertRaisesRegex(ValueError,"Exactly agent1 and agent2"):
            agent_view(fixture)
    def test_mock(self):
        a=ScriptedAgent([{"action":{"tool":"bash","command":"echo hi"},"files_touched":[]}])
        e=MockEnvironment()
        a.receive(e.execute("agent1",a.propose({})["action"]))
        self.assertIsNone(a.propose({}))
        self.assertEqual(len(e.executed),1)
        e.close()
        with self.assertRaises(RuntimeError):
            e.execute("agent1",{})

    def test_installed_cooperbench_revision(self):
        pin=Path("runner/requirements-cooperbench.txt").read_text().strip()
        repo, revision=pin.removeprefix("git+").rsplit("@",1)
        dist=Mock()
        source={"url":repo,"vcs_info":{"vcs":"git","commit_id":revision}}
        with patch("runner.cooperbench_adapter.metadata.distribution",return_value=dist):
            dist.read_text.return_value=json.dumps(source)
            self.assertEqual(installed_revision(),revision)
            source["vcs_info"]["commit_id"]="0"*40 if revision!="0"*40 else "1"*40
            dist.read_text.return_value=json.dumps(source)
            with self.assertRaisesRegex(RuntimeError,"does not match"):
                installed_revision()
            source["vcs_info"]["commit_id"]=revision
            source["url"]="https://example.com/unrelated.git"
            dist.read_text.return_value=json.dumps(source)
            with self.assertRaisesRegex(RuntimeError,"does not match"):
                installed_revision()
            dist.read_text.return_value=None
            with self.assertRaisesRegex(RuntimeError,"does not match"):
                installed_revision()

    def test_cooperbench_missing_installation(self):
        with patch("runner.cooperbench_adapter.metadata.distribution",
                   side_effect=metadata.PackageNotFoundError("cooperbench")):
            with self.assertRaisesRegex(RuntimeError,"does not match"):
                installed_revision()

    def test_dependency_pin_file_requires_one_valid_revision(self):
        revision="a"*40
        pin=f"git+https://github.com/cooperbench/CooperBench.git@{revision}"
        for content in ("\n# missing pin\nother==1\n", f"{pin}\n{pin}\n"):
            with self.subTest(content=content), patch("runner.cooperbench_adapter.Path.read_text",return_value=content), \
                 patch("runner.cooperbench_adapter.metadata.distribution") as distribution:
                with self.assertRaisesRegex(RuntimeError,"exactly one"):
                    installed_revision()
                distribution.assert_not_called()

    def test_preparation_uses_declared_base_and_timeout(self):
        task={"dataset_source":"cooperbench","base_commit":"a"*40,
              "metadata":{"cooperbench_repo_directory":"fixture","cooperbench_task_id":1,"feature_ids":[1,2]},
              "goals":[{"goal_key":"feature1","owner_agent_id":"agent1"},
                       {"goal_key":"feature2","owner_agent_id":"agent2"}]}
        with patch("runner.cooperbench_adapter.DockerEnvironment") as docker:
            self.assertIs(prepare_environment(task,"image",timeout=3),docker.return_value)
            docker.assert_called_once_with("image",task["base_commit"],timeout=3)
        with patch("runner.cooperbench_adapter.SharedDockerEnvironment") as shared, \
                patch("runner.cooperbench_adapter.image_id", return_value="fixture-id") as resolve:
            env = prepare_environment(task, "image", timeout=9, concurrent=True)
        deadline = resolve.call_args.args[1]
        shared.assert_called_once_with("fixture-id", task["base_commit"], timeout=9, deadline=deadline)
        self.assertEqual(env.image_id, "fixture-id")
