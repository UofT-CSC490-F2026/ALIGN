import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from runner.cli import main
from scripts.run_artifacts import verify_bundle,load_events
from runner.model_adapter import SafetyRuleArbiter

class RunnerIntegrationTests(unittest.TestCase):
    def test_mock_cli(self):
        from scripts.normalize_tasks import normalized_custom,write_output
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            scenario=Path("scripts/examples")
            records=normalized_custom(scenario,"v1")
            write_output(records,root/"tasks","custom","v1","all")
            code=main(["--tasks",str(root/"tasks"/"tasks.jsonl"),"--output-dir",str(root/"runs"),
                       "--condition","llm","--mock-action","block"])
            self.assertEqual(code,0)
            bundle=next((root/"runs").glob("*/bundle"))
            self.assertTrue(verify_bundle(bundle)["complete"])
            events,_=load_events((bundle/"events.jsonl").read_bytes())
            self.assertIsNone(events[0]["payload"]["agent_config"]["upstream_revision"])
            self.assertTrue(any(e["event_type"]=="arbiter_decision" for e in events))
    def test_rule_baseline(self):
        d=SafetyRuleArbiter().decide({"actor_id":"agent1","constraints":{},
           "proposed_action":{"action":{"command":"git push --force"},"files_touched":[]}})
        self.assertEqual(d["action"],"block")

    def test_empty_constraints_vacuous(self):
        from runner.cooperbench_adapter import evaluation_evidence
        task={"dataset_source":"cooperbench","constraints":{},
              "metadata":{"cooperbench_repo_directory":"repo_task","cooperbench_task_id":1,"feature_ids":[1,2]},
              "goals":[{"goal_key":f"feature{i}","owner_agent_id":f"agent{i}"} for i in (1,2)]}
        result={"feature1":{"passed":True},"feature2":{"passed":True},
                "both_passed":True,"merge":{"status":"clean"}}
        self.assertTrue(evaluation_evidence(task,result)["critical_constraints_passed"])
        task["constraints"]={"protected_paths":["tests/"]}
        self.assertIsNone(evaluation_evidence(task,result)["critical_constraints_passed"])
