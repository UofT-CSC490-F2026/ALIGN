import unittest
import json
from pathlib import Path
from unittest.mock import patch, Mock
from importlib import metadata
from runner.cooperbench_adapter import installed_revision
from runner.environment import MockEnvironment, ScriptedAgent
from runner.task_loader import agent_view

class EnvironmentTests(unittest.TestCase):
    def test_projection(self):
        t={"task_key":"x","shared_goal":"both","constraints":{"secret":"hidden"},
           "goals":[{"owner_type":"agent","owner_agent_id":f"agent{i}","description":"edit",
                     "success_criteria":{"secret":"hidden"}} for i in (1,2)]}
        self.assertNotIn("hidden",str(agent_view(t)))
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
