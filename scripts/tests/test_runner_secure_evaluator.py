import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from runner.cooperbench_adapter import evaluate_patches, evaluation_evidence
from runner.deadline import TrialDeadline
from runner.docker_utils import OutputLimitExceeded
from runner.secure_evaluator import SecureSandbox
import subprocess


TASK = {"dataset_source":"cooperbench", "base_commit":"b"*40, "constraints":{},
        "metadata":{"cooperbench_repo_directory":"fixture_task", "cooperbench_task_id":0,
                    "feature_ids":[1,9]},
        "goals":[{"goal_key":"feature1", "owner_agent_id":"agent1"},
                 {"goal_key":"feature9", "owner_agent_id":"agent2"}]}
IMAGE = "sha256:"+"a"*64
PASSED = {"passed":True,"exit_code":0,"tests_passed":1,"tests_failed":0}


def upstream_modules(sandbox):
    package = ModuleType("cooperbench.eval")
    package.sandbox = sandbox
    root = ModuleType("cooperbench")
    root.eval = package
    return {"cooperbench":root, "cooperbench.eval":package}


class SecureEvaluationTests(unittest.TestCase):
    def test_each_feature_uses_clean_sandbox_and_first_pass_survives_timeout(self):
        upstream = SimpleNamespace(_filter_test_files=lambda text:text,
                                   _run_tests=Mock(side_effect=[PASSED,TimeoutError()]))
        first, second = Mock(), Mock()
        with TemporaryDirectory() as tmp:
            for feature in (1,9):
                path = Path(tmp)/"fixture_task"/"task0"/f"feature{feature}"
                path.mkdir(parents=True)
                (path/"tests.patch").write_text(f"tests for {feature}",encoding="utf-8")
            with patch.dict(sys.modules,upstream_modules(upstream)), \
                 patch("runner.cooperbench_adapter.image_id",return_value=IMAGE), \
                 patch("runner.cooperbench_adapter.SecureBackend") as backend:
                backend.return_value.create_sandbox.side_effect = [first,second]
                result = evaluate_patches(TASK,{"_merge_status":"clean","agent1":"merged"},tmp,
                                          expected_image=IMAGE)
        self.assertEqual(result["goal_results"],{"feature1":"passed","feature9":"not_evaluated"})
        self.assertTrue(result["evaluation_timeout"])
        self.assertIsNone(result["shared_goal_satisfied"])
        first.terminate.assert_called_once()
        second.terminate.assert_called_once()
        first.write_patch.assert_any_call("tests1.patch","tests for 1")
        second.write_patch.assert_any_call("tests2.patch","tests for 9")

    def test_conflicts_skip_tests_and_identity_mismatch_is_rejected(self):
        upstream = SimpleNamespace(get_image_name=Mock(return_value="click"))
        with patch.dict(sys.modules,upstream_modules(upstream)), \
             patch("runner.cooperbench_adapter.image_id", return_value=IMAGE), \
             patch("runner.cooperbench_adapter.SecureBackend") as backend:
            result = evaluate_patches(TASK,{"_merge_status":"conflicts"},"unused")
            self.assertEqual(result["goal_results"],{"feature1":"not_evaluated","feature9":"not_evaluated"})
            self.assertFalse(result["shared_goal_satisfied"])
            upstream.get_image_name.assert_called_once_with("fixture_task",0)
            with self.assertRaisesRegex(ValueError,"image IDs differ"):
                evaluate_patches(TASK,{},"unused",expected_image="sha256:"+"b"*64)
            backend.assert_not_called()

    def test_legacy_evaluation_injects_local_backend_and_preserves_cleanup_evidence(self):
        def test_merged(repo, number, first, second, *, patch1, patch2, timeout, backend, dataset_dir):
            factory = globals()["get_backend"](backend)
            sandbox = factory.create_sandbox(globals()["get_image_name"](repo, number), timeout)
            globals()["_write_patch"](sandbox,"first.patch",patch1.read_text(encoding="utf-8"))
            globals()["_write_patch"](sandbox,"second.patch",patch2.read_text(encoding="utf-8"))
            return {"merge":{"status":"clean"},"both_passed":True,
                    "feature1":{**PASSED,"feature_id":first},"feature2":{**PASSED,"feature_id":second}}

        with patch.dict(sys.modules,upstream_modules(SimpleNamespace(test_merged=test_merged))), \
             patch("runner.cooperbench_adapter.image_id",return_value=IMAGE), \
             patch("runner.cooperbench_adapter.SecureBackend") as factory:
            backend = factory.return_value
            backend.close.side_effect = RuntimeError("could not remove sandbox")
            result = evaluate_patches(TASK,{"agent1":"first patch","agent2":"second patch"},"unused",
                                      expected_image=IMAGE)
        backend.create_sandbox.assert_called_once_with(IMAGE, 600)
        sb = backend.create_sandbox.return_value
        sb.write_patch.assert_any_call("first.patch","first patch")
        sb.write_patch.assert_any_call("second.patch","second patch")
        self.assertEqual(result["goal_results"],{"feature1":"passed","feature9":"passed"})
        self.assertTrue(result["evaluation_error"])
        self.assertIsNone(result["shared_goal_satisfied"])

    def test_truncated_evaluator_output_is_not_scored(self):
        sandbox = SecureSandbox.__new__(SecureSandbox)
        sandbox.deadline = TrialDeadline(10)
        sandbox.timeout = 5
        sandbox.closed = False
        sandbox.workdir = "/workspace"
        sandbox.name = "fixture"
        result = subprocess.CompletedProcess([], 0, b"1 passed", b"")
        result.output_truncated = True
        with patch("runner.secure_evaluator.bounded_run", return_value=result):
            with self.assertRaises(OutputLimitExceeded) as error:
                sandbox.exec("test-command")
        self.assertEqual(error.exception.returncode, 0)

    def test_adapter_records_complete_goals_and_handles_output_limit(self):
        with TemporaryDirectory() as tmp:
            for feature in (1,9):
                path=Path(tmp)/"fixture_task"/"task0"/f"feature{feature}"
                path.mkdir(parents=True)
                (path/"tests.patch").write_text("fixture",encoding="utf-8")
            for second in (PASSED,OutputLimitExceeded(17)):
                upstream=SimpleNamespace(_filter_test_files=lambda text:text,
                                         _run_tests=Mock(side_effect=[PASSED,second]))
                with self.subTest(second=second), patch.dict(sys.modules,upstream_modules(upstream)), \
                     patch("runner.cooperbench_adapter.image_id",return_value=IMAGE), \
                     patch("runner.cooperbench_adapter.SecureBackend"):
                    result=evaluate_patches(TASK,{"_merge_status":"clean","agent1":"merged"},tmp,
                                            expected_image=IMAGE)
                    limited=isinstance(second,OutputLimitExceeded)
                    self.assertEqual(result["goal_results"],{"feature1":"passed",
                                                              "feature9":"not_evaluated" if limited else "passed"})
                    self.assertEqual(result["evaluation_error"],limited)
                    self.assertFalse(result["evaluation_timeout"])
                    self.assertEqual(result["evaluation_output_truncated"],limited)
                    self.assertEqual(result["evaluation_exit_code"],17 if limited else None)
                    self.assertIs(result["shared_goal_satisfied"],None if limited else True)

    def test_missing_joint_evidence_does_not_overwrite_observed_goals(self):
        evidence=evaluation_evidence(TASK,{"merge":{"status":"clean"},"feature1":PASSED,"feature2":PASSED})
        self.assertEqual(evidence["goal_results"],{"feature1":"passed","feature9":"passed"})
        self.assertIsNone(evidence["shared_goal_satisfied"])

    def test_deadline_after_upstream_result_retains_tests_without_claiming_success(self):
        deadline=TrialDeadline(30)

        def test_merged(*args,**kwargs):
            deadline.cancel()
            return {"merge":{"status":"clean"},"both_passed":True,"feature1":PASSED,"feature2":PASSED}

        with patch.dict(sys.modules,upstream_modules(SimpleNamespace(test_merged=test_merged))), \
             patch("runner.cooperbench_adapter.image_id",return_value=IMAGE), \
             patch("runner.cooperbench_adapter.SecureBackend"):
            result=evaluate_patches(TASK,{"agent1":"first","agent2":"second"},"unused",
                                    expected_image=IMAGE,deadline=deadline)
        self.assertEqual(result["goal_results"],{"feature1":"passed","feature9":"passed"})
        self.assertTrue(result["evaluation_timeout"])
        self.assertIsNone(result["shared_goal_satisfied"])
