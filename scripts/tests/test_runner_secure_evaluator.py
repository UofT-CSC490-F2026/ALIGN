import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from runner.cooperbench_adapter import evaluate_patches


class SecureEvaluationTests(unittest.TestCase):
    def test_each_feature_uses_clean_sandbox_and_first_pass_survives_timeout(self):
        task = {"dataset_source":"cooperbench", "base_commit":"b"*40, "constraints":{},
                "metadata":{"cooperbench_repo_directory":"fixture_task","cooperbench_task_id":0,
                            "feature_ids":[1,9]},
                "goals":[{"goal_key":"feature1","owner_agent_id":"agent1"},
                         {"goal_key":"feature9","owner_agent_id":"agent2"}]}
        image = "sha256:"+"a"*64
        passed = {"passed":True,"exit_code":0,"tests_passed":1,"tests_failed":0}
        upstream = SimpleNamespace(_filter_test_files=lambda text:text,
                                   _run_tests=Mock(side_effect=[passed,TimeoutError()]))
        package = ModuleType("cooperbench.eval")
        package.sandbox = upstream
        root = ModuleType("cooperbench")
        root.eval = package
        first, second = Mock(), Mock()
        with TemporaryDirectory() as tmp:
            for feature in (1,9):
                path = Path(tmp)/"fixture_task"/"task0"/f"feature{feature}"
                path.mkdir(parents=True)
                (path/"tests.patch").write_text(f"tests for {feature}",encoding="utf-8")
            with patch.dict(sys.modules,{"cooperbench":root,"cooperbench.eval":package}), \
                 patch("runner.cooperbench_adapter.image_id",return_value=image), \
                 patch("runner.cooperbench_adapter.SecureBackend") as backend:
                backend.return_value.create_sandbox.side_effect = [first,second]
                result = evaluate_patches(task,{"_merge_status":"clean","agent1":"merged"},tmp,
                                          expected_image=image)
        self.assertEqual(result["goal_results"],{"feature1":"passed","feature9":"not_evaluated"})
        self.assertTrue(result["evaluation_timeout"])
        self.assertIsNone(result["shared_goal_satisfied"])
        first.terminate.assert_called_once()
        second.terminate.assert_called_once()
        first.write_patch.assert_any_call("tests1.patch","tests for 1")
        second.write_patch.assert_any_call("tests2.patch","tests for 9")
