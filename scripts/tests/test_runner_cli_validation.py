import contextlib
import builtins
import hashlib
import io
import json
import runpy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import Mock, patch

from runner.cli import main
from runner.environment import MockEnvironment
from runner.task_loader import cooperbench_identity, load_tasks
from scripts.run_artifacts import load_events
from test_runner_orchestration import task


class RunnerCliValidationTests(unittest.TestCase):
    def test_rejects_invalid_configuration_before_starting_trial(self):
        live = ["--live", "--agent-endpoint", "http://localhost:1/v1", "--agent-model", "test",
                "--images", "missing.json", "--dataset-dir", "missing-dataset"]
        cases = (
            (["--task-key", "missing"], "Unknown task key"),
            (["--limit", "0"], "Positive limits required"),
            (["--max-steps", "0"], "Positive limits required"),
            (["--trial-timeout", "inf"], "finite and positive"),
            (["--live"], "Live mode requires"),
            (live + ["--protocol", "alternating"], "Live experiments require"),
            (live + ["--condition", "monitor", "--arbiter-model", "qwen"], "requires an arbiter model"),
        )
        with TemporaryDirectory() as root, patch("runner.cli.load_tasks", return_value=[task()]), \
             patch("runner.cli.installed_revision") as revision:
            out = Path(root) / "runs"
            for flags, message in cases:
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, message):
                    main(["--tasks", "fixture", "--output-dir", str(out), *flags])
            self.assertFalse(out.exists())
            revision.assert_not_called()

    def test_failed_initialization_emits_sanitized_complete_bundle(self):
        fixture = task()
        with TemporaryDirectory() as root:
            images = Path(root) / "images.json"
            images.write_text('{"sample_task:task1":"local"}', encoding="utf-8")
            with patch("runner.cli.load_tasks", return_value=[fixture]), \
                 patch("runner.cli.cooperbench_identity", return_value=("sample_task", 1, [1, 2])), \
                 patch("runner.cli.installed_revision", return_value="a" * 40), \
                 patch("runner.cli.prepare_environment", side_effect=TimeoutError("private details")), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = main(["--tasks", "fixture", "--output-dir", root, "--live", "--agent-endpoint",
                             "http://localhost:1/v1", "--agent-model", "test", "--images", str(images),
                             "--dataset-dir", root])
            self.assertEqual(code, 1)
            raw = next(Path(root).glob("*/bundle/events.jsonl")).read_bytes()
            events, _ = load_events(raw)
            self.assertEqual([e["event_type"] for e in events], ["run_start", "run_end"])
            self.assertEqual(events[-1]["payload"]["failure_category"], "timeout")
            self.assertNotIn(b"private details", raw)

    def test_selected_rule_based_run_closes_environment_on_unexpected_error(self):
        fixture=task()
        other={**fixture,"task_key":"other"}
        env=Mock()
        env.close.side_effect=OSError("private cleanup info")
        with TemporaryDirectory() as root, patch("runner.cli.load_tasks",return_value=[other,fixture]), \
             patch("runner.cli.MockEnvironment",return_value=env), \
             patch("runner.cli.run_task",side_effect=RuntimeError("private runner info")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code=main(["--tasks","fixture","--output-dir",root,"--task-key",fixture["task_key"],
                       "--condition","rule_based"])
            self.assertEqual(code,1)
            self.assertEqual(json.loads(output.getvalue())["task_key"],fixture["task_key"])
            events,header=load_events(next(Path(root).glob("*/bundle/events.jsonl")).read_bytes())
        env.close.assert_called_once()
        self.assertEqual(header["arbiter_config"],{"rules":"a2-provisional-v1"})
        self.assertEqual(events[-1]["payload"]["failure_category"],"infrastructure_error")
        self.assertNotIn("private",str(events))

    def test_agents_with_unknown_usage_never_publish_zero_tokens(self):
        with TemporaryDirectory() as root:
            images=Path(root)/"images.json"
            images.write_text('{"sample_task:task1":"local"}',encoding="utf-8")
            client=Mock(usage_known=False)
            client.generate.return_value={"done":True}
            with patch("runner.cli.load_tasks",return_value=[task()]), \
                 patch("runner.cli.cooperbench_identity",return_value=("sample_task",1,[1,2])), \
                 patch("runner.cli.installed_revision",return_value="a"*40), \
                 patch("runner.cli.ChatClient",return_value=client), \
                 patch("runner.cli.prepare_environment",return_value=MockEnvironment()), \
                 patch("runner.cli.evaluate_patches",return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--tasks","fixture","--output-dir",root,"--live","--agent-endpoint",
                                       "http://localhost:1/v1","--agent-model","test","--images",str(images),
                                       "--dataset-dir",root]),0)
            events,_=load_events(next(Path(root).glob("*/bundle/events.jsonl")).read_bytes())
        self.assertTrue(events[-1]["payload"]["usage_incomplete"])
        self.assertNotIn("input_tokens",events[-1]["payload"])

    def test_terminal_event_is_preserved_if_runner_errors_after_finishing(self):
        def finished(task,agents,env,event_writer,**kwargs):
            event_writer.finish("failed",failure_category="timeout",failure_reason="Original failure")
            raise RuntimeError("later failure")

        with TemporaryDirectory() as root, patch("runner.cli.load_tasks",return_value=[task()]), \
             patch("runner.cli.run_task",side_effect=finished), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--tasks","fixture","--output-dir",root]),1)
            events,_=load_events(next(Path(root).glob("*/bundle/events.jsonl")).read_bytes())
        self.assertEqual([e["event_type"] for e in events],["run_start","run_end"])
        self.assertEqual(events[-1]["payload"]["failure_reason"],"Original failure")

    def test_modal_validation_and_callable_factory_without_remote_calls(self):
        flags = ["--tasks", "fixture", "--output-dir", "unused", "--arbiter-backend", "modal"]
        for options, error in (([], "requires --live"),
                               (["--live", "--condition", "llm"], "requires --arbiter-model"),
                               (["--live", "--condition", "llm", "--arbiter-model", "qwen", "--training", "sft"],
                                "requires --checkpoint")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as exit:
                    main(flags + options)
                self.assertEqual(exit.exception.code, 2)
                self.assertIn(error, stderr.getvalue())

        inference = ModuleType("arbiter_training.inference")
        inference.ModalArbiter = Mock()
        modal = ModuleType("arbiter_training.modal_app")
        modal.app = Mock()
        modal.app.run.return_value = contextlib.nullcontext()

        def run(args, *, arbiter_factory):
            self.assertIs(arbiter_factory(args), inference.ModalArbiter.return_value)
            return 0

        for checkpoint in (None, "adapter-v1"):
            options = ["--training", "sft", "--checkpoint", checkpoint] if checkpoint else []
            with patch.dict(sys.modules, {inference.__name__:inference, modal.__name__:modal}), \
                 patch("runner.cli.run", side_effect=run):
                self.assertEqual(main(flags + ["--live", "--condition", "llm", "--arbiter-model", "qwen"] + options), 0)
            inference.ModalArbiter.assert_called_with(model="qwen", adapter_path=checkpoint or "", invalid_policy="raise")

    def test_runner_module_help(self):
        with patch.object(sys, "argv", ["runner.cli", "--help"]), \
             patch.dict(sys.modules), contextlib.redirect_stdout(io.StringIO()) as help_text:
            sys.modules.pop("runner.cli", None)
            with self.assertRaises(SystemExit) as exit:
                runpy.run_module("runner.cli", run_name="__main__")
        self.assertEqual(exit.exception.code, 0)
        self.assertIn("--trial-timeout", help_text.getvalue())


class TaskIdentityTests(unittest.TestCase):
    def test_empty_or_duplicate_input_fails_before_validation(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "tasks.jsonl"
            for text in ("", (json.dumps({"task_key":"duplicate"}) + "\n") * 2):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Empty or duplicate tasks"):
                    load_tasks(path)

    def test_invalid_cooperbench_identity_and_swapped_goal_owners(self):
        valid = {"dataset_source":"cooperbench", "metadata":{"cooperbench_repo_directory":"sample_task",
                 "cooperbench_task_id":0, "feature_ids":[1, 2]},
                 "goals":[{"goal_key":"feature1", "owner_agent_id":"agent1"},
                          {"goal_key":"feature2", "owner_agent_id":"agent2"}]}
        for key, value in (("cooperbench_repo_directory", ".."), ("cooperbench_repo_directory", "test/repo"),
                           ("cooperbench_repo_directory", "test\\repo"), ("cooperbench_task_id", -1),
                           ("cooperbench_task_id", True), ("feature_ids", [2, 1]),
                           ("feature_ids", [0, 2]), ("feature_ids", [1])):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "Invalid CooperBench identity"):
                cooperbench_identity({**valid, "metadata":{**valid["metadata"], key:value}})
        with self.assertRaisesRegex(ValueError, "Not a CooperBench task"):
            cooperbench_identity({**valid, "dataset_source":"custom"})
        valid["goals"][0]["owner_agent_id"] = "agent2"
        with self.assertRaisesRegex(ValueError, "Feature assignment mismatch"):
            cooperbench_identity(valid)

    def test_prompt_falls_back_only_when_training_package_is_missing(self):
        original_import=builtins.__import__

        def load_prompt(missing):
            def resolve(name,*args,**kwargs):
                if name=="arbiter_training.contract":
                    raise ModuleNotFoundError(name=missing)
                return original_import(name,*args,**kwargs)
            with patch("builtins.__import__",side_effect=resolve):
                return runpy.run_path("runner/arbiter_prompt.py")

        fallback=load_prompt("arbiter_training")
        self.assertEqual(fallback["prompt_hash"](),hashlib.sha256(fallback["SYSTEM_PROMPT"].encode()).hexdigest())
        with self.assertRaises(ModuleNotFoundError) as error:
            load_prompt("unrelated_dependency")
        self.assertEqual(error.exception.name,"unrelated_dependency")
