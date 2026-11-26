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

    def test_cooperbench_conflict_does_not_invent_test_failure(self):
        from runner.cooperbench_adapter import evaluation_evidence
        task={"dataset_source":"cooperbench","constraints":{},
              "metadata":{"cooperbench_repo_directory":"repo_task","cooperbench_task_id":1,"feature_ids":[3,7]},
              "goals":[{"goal_key":f"feature{f}","owner_agent_id":f"agent{i}"} for i,f in enumerate((3,7),1)]}
        result={"feature1":{"passed":False,"exit_code":None,"feature_id":3},
                "feature2":{"passed":False,"exit_code":None,"feature_id":7},
                "both_passed":False,"merge":{"status":"conflicts"}}
        evidence=evaluation_evidence(task,result)
        self.assertEqual(evidence["goal_results"],{"feature3":"not_evaluated","feature7":"not_evaluated"})
        self.assertIs(evidence["shared_goal_satisfied"],False)
        # A runner failure before tests start also has no goal evidence.
        result["merge"]["status"]="clean"
        result["feature1"].update(exit_code=1,tests_passed=0,tests_failed=0)
        evidence=evaluation_evidence(task,result)
        self.assertEqual(evidence["goal_results"]["feature3"],"not_evaluated")
        self.assertIsNone(evidence["shared_goal_satisfied"])
        self.assertTrue(evidence["evaluation_error"])

    def test_pinned_flash_includes_task_zero(self):
        from runner.task_loader import cooperbench_identity
        task={"dataset_source":"cooperbench",
              "metadata":{"cooperbench_repo_directory":"openai_tiktoken_task","cooperbench_task_id":0,"feature_ids":[2,9]},
              "goals":[{"goal_key":f"feature{f}","owner_agent_id":f"agent{i}"} for i,f in enumerate((2,9),1)]}
        self.assertEqual(cooperbench_identity(task),("openai_tiktoken_task",0,[2,9]))
        task["metadata"]["cooperbench_task_id"]=-1
        with self.assertRaises(ValueError):
            cooperbench_identity(task)
