import unittest
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
