"""Exercise Modal entry points with local fakes; no GPU or network is used."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")

from arbiter_training import contract, modal_app


def observation():
    return {
        "schema_version": 1, "task_key": "cooperbench:train:task1:f1_f2",
        "shared_goal": "Both features", "agent_goals": {"agent1": "First", "agent2": "Second"},
        "constraints": {}, "actor_id": "agent1", "proposal_sequence": 2,
        "proposed_action": {"action": {"tool": "bash", "command": "git status"},
                            "files_touched": []},
        "recent_events": [], "repository_state": {},
    }


def record(repo="train", action="block"):
    item = observation()
    item["task_key"] = f"cooperbench:{repo}:task1:f1_f2"
    decision = contract.decision_for_observation(action, "Reason", item)
    return {"task_key": item["task_key"], "primary_action": action,
            "expected_actions": [action], "label_source": "human",
            "review_status": "reviewed", "scenario_source": "public_trace",
            "checkpoint_event_uuid": f"{repo}-{action}",
            "prompt": contract.messages_for_observation(item),
            "completion": [{"role": "assistant", "content": json.dumps(decision)}]}


def test_modal_model_loading_and_generation_are_offline(monkeypatch):
    loaded = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            loaded.append((messages, kwargs))
            return "PROMPT"
        def encode(self, prompt, add_special_tokens=False):
            return list(prompt)
    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def get_tokenizer(self):
            return Tokenizer()
        def generate(self, prompts, params, lora_request=None):
            loaded.append((prompts, params, lora_request))
            return [SimpleNamespace(outputs=[SimpleNamespace(text="{}", finish_reason="stop")])
                    for _ in prompts]
    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class LoRARequest:
        def __init__(self, *args):
            self.args = args
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    monkeypatch.setitem(sys.modules, "vllm.lora", SimpleNamespace(request=SimpleNamespace(LoRARequest=LoRARequest)))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=LoRARequest))
    with pytest.raises(ValueError, match="Unknown model"):
        modal_app._load_inference_engine("unknown", "")
    with pytest.raises(ValueError, match="Only the Qwen"):
        modal_app._load_inference_engine("gpt-oss", "/adapter")
    qwen, adapter = modal_app._load_inference_engine("qwen", "/adapter")
    assert qwen.kwargs["enable_lora"] is True
    assert adapter.args == ("align-arbiter", 1, "/adapter")
    assert modal_app._generate_batch(qwen, adapter, "qwen", "/adapter", [[{"role": "user", "content": "Hi"}]])[0]["adapter"] == "/adapter"
    assert loaded[0][1]["enable_thinking"] is False
    gpt, no_adapter = modal_app._load_inference_engine("gpt-oss", "")
    assert no_adapter is None
    output = modal_app._generate_batch(gpt, None, "gpt-oss", "", [[{"role": "user", "content": "Hi"}]])
    assert output[0]["adapter"] is None
    assert loaded[-1][0][0].endswith("<|channel|>final<|message|>")
    class Oversized:
        def encode(self, prompt, add_special_tokens=False):
            return range(modal_app.MAX_MODEL_LEN)
    with pytest.raises(ValueError, match="context exceeds"):
        modal_app._require_context_fit(Oversized(), ["x"])
    assert modal_app._require_context_fit(Oversized(), [], reserve=0) == []
    assert modal_app.infer_batch._raw_f_("qwen", [[{"role": "user", "content": "Hi"}]])
    assert modal_app._decision_text("gpt-oss", "assistantfinal json {\"action\": 1}") == '{"action": 1}'
    assert modal_app._decision_text("qwen", "plain") == "plain"


def test_warm_modal_class_delegates_to_local_engine(monkeypatch):
    user_class = modal_app.ArbiterModel._get_user_cls()
    instance = SimpleNamespace(model_name="qwen", adapter_path="/adapter")
    monkeypatch.setattr(modal_app, "_load_inference_engine", lambda model, adapter: ("engine", "lora"))
    user_class.setup._get_raw_f()(instance)
    assert (instance.llm, instance.lora_request) == ("engine", "lora")
    monkeypatch.setattr(modal_app, "_generate_batch",
                        lambda *args: {"engine": args[0], "model": args[2], "batch": args[4]})
    assert user_class.decide_batch._get_raw_f()(instance, [[{"role": "user", "content": "x"}]])["engine"] == "engine"


def test_modal_validation_helpers(tmp_path):
    file = tmp_path / "rows.jsonl"
    file.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")
    assert list(modal_app._records(file)) == [{"x": 1}, {"x": 2}]
    modal_app._require_runner_contract([record()])
    bad = record()
    item = json.loads(bad["prompt"][1]["content"])
    item["proposed_action"] = {"action": "wrong", "files_touched": []}
    bad["prompt"][1]["content"] = json.dumps(item)
    with pytest.raises(ValueError, match="ProposedAction"):
        modal_app._require_runner_contract([bad])


def test_training_input_guards_and_mocked_sft(monkeypatch):
    train = [record("train", "block") for _ in range(20)]
    val = [record("val", "continue") for _ in range(5)]
    monkeypatch.setattr(modal_app, "_records", lambda path: iter(train if "train" in str(path) else val))
    raw = modal_app.train_qwen._raw_f_
    with pytest.raises(ValueError, match="run_name"):
        raw("bad/name")
    with pytest.raises(ValueError, match="max_steps"):
        raw("good", 0)
    with pytest.raises(ValueError, match="inside /data"):
        raw("good", 1, "/elsewhere/train.jsonl")
    monkeypatch.setattr(modal_app, "_records", lambda path: iter([]))
    with pytest.raises(ValueError, match="at least 20"):
        raw("good", 1)
    monkeypatch.setattr(modal_app, "_records", lambda path: iter(train if "train" in str(path) else val))
    train[0]["prompt"][1]["content"] = json.dumps({"proposed_action": {"action": "bad", "files_touched": []}})
    with pytest.raises(ValueError, match="ProposedAction"):
        raw("good", 1)
    train[0] = record("train", "block")
    for row in train:
        row["primary_action"] = "continue"
    with pytest.raises(ValueError, match="intervention"):
        raw("good", 1)
    train[:] = [record("train", "block") for _ in range(20)]
    val[0]["task_key"] = train[0]["task_key"]
    with pytest.raises(ValueError, match="tasks overlap"):
        raw("good", 1)
    val[0] = record("val", "continue")
    val[0]["task_key"] = "cooperbench:train:task2:f1_f2"
    with pytest.raises(ValueError, match="repositories overlap"):
        raw("good", 1)
    val[0] = record("val", "continue")
    val[0]["label_source"] = "unknown"
    with pytest.raises(ValueError, match="provenance"):
        raw("good", 1)
    val[0] = record("val", "continue")
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: True if str(self).endswith("already") else False)
    with pytest.raises(ValueError, match="already exists"):
        raw("already", 1)
    monkeypatch.setattr(Path, "exists", lambda self: False)

    calls = []
    class Tokenizer:
        eos_token = "<eos>"
        def apply_chat_template(self, prompt, **kwargs):
            calls.append(("template", kwargs))
            return "prompt"
        def encode(self, prompt, add_special_tokens=False):
            return list(prompt)
        def save_pretrained(self, path):
            calls.append(("tokenizer", path))
    class Model:
        def save_pretrained(self, path):
            calls.append(("adapter", path))
    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(("base", kwargs))
            return Model(), Tokenizer()
        @staticmethod
        def get_peft_model(model, **kwargs):
            calls.append(("peft", kwargs))
            return model
    class Trainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.state = SimpleNamespace(best_metric=.5)
            calls.append(("trainer", kwargs))
        def train(self):
            calls.append(("train", None))
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=FastLanguageModel))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)))
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(SFTConfig=lambda **kw: kw, SFTTrainer=Trainer))
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"pinned data")
    saved = []
    monkeypatch.setattr(Path, "write_text", lambda self, text, **kw: saved.append((str(self), json.loads(text))))
    monkeypatch.setattr(modal_app, "checkpoint_volume", SimpleNamespace(commit=lambda: calls.append(("commit", None))))
    metadata = raw("good", 4)
    assert metadata["train_examples"] == 20 and metadata["validation_examples"] == 5
    assert metadata["best_metric"] == .5
    assert metadata["max_train_tokens"] > 0
    assert any(name == "adapter" for name, _ in calls)
    assert saved[0][1]["run_name"] == "good"
    assert calls[-1][0] == "commit"


def test_modal_main_modes_without_remote_calls(tmp_path, monkeypatch, capsys):
    decision = contract.decision_for_observation("continue", "Safe", observation())
    response = {"text": json.dumps(decision)}
    calls = []
    monkeypatch.setattr(modal_app, "infer_batch", SimpleNamespace(remote=lambda *a: calls.append(a) or [dict(response)]))
    monkeypatch.setattr(modal_app, "train_qwen", SimpleNamespace(remote=lambda *a: {"trained": a[0]}))
    with pytest.raises(ValueError, match="run-name"):
        modal_app.main(mode="train")
    modal_app.main(mode="train", run_name="job", steps=2)
    assert json.loads(capsys.readouterr().out)["trained"] == "job"
    with pytest.raises(ValueError, match="checkpoint-file"):
        modal_app.main(mode="infer")
    checkpoint = tmp_path / "observation.json"
    checkpoint.write_text(json.dumps(observation()), encoding="utf-8")
    modal_app.main(mode="infer", checkpoint_file=str(checkpoint))
    assert json.loads(capsys.readouterr().out)["decision"] == decision
    assert calls[-1][0] == "qwen"
    wrapped = {"schema_version": 1, "task": {
        "task_key": observation()["task_key"], "shared_goal": "Both features",
        "goals": [{"owner_agent_id": "agent1", "description": "First"},
                  {"owner_agent_id": "agent2", "description": "Second"}], "constraints": {}},
        "agent_id": "agent1", "sequence_number": 2,
        "proposal": observation()["proposed_action"], "prior_events": []}
    checkpoint.write_text(json.dumps(wrapped), encoding="utf-8")
    modal_app.main(mode="infer", checkpoint_file=str(checkpoint))
    assert json.loads(capsys.readouterr().out)["decision"] == decision
    monkeypatch.setattr(modal_app, "infer_batch", SimpleNamespace(remote=lambda *a: [{"text": "invalid"}]))
    modal_app.main(mode="infer", checkpoint_file=str(checkpoint))
    assert "parse_error" in json.loads(capsys.readouterr().out)
    with pytest.raises(ValueError, match="validation-file"):
        modal_app.main(mode="class-smoke")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    import arbiter_training.inference as inference
    monkeypatch.setattr(inference, "ModalArbiter", lambda **kw: SimpleNamespace(decide=lambda obs: decision))
    monkeypatch.setattr(modal_app, "ArbiterModel", lambda **kw: object())
    modal_app.main(mode="class-smoke", validation_file=str(validation))
    assert json.loads(capsys.readouterr().out)["decision"] == decision
    with pytest.raises(ValueError, match="Use --mode"):
        modal_app.main(mode="wrong")


def test_modal_comparison_report_and_rejections(tmp_path, monkeypatch, capsys):
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    decision = contract.decision_for_observation("block", "Reason", observation())
    monkeypatch.setattr(modal_app, "infer_batch", SimpleNamespace(remote=lambda *a: [{"text": json.dumps(decision)}]))
    report = tmp_path / "report.json"
    modal_app.main(mode="compare", validation_file=str(validation), models="qwen", output_file=str(report))
    assert json.loads(capsys.readouterr().out)["models"]["qwen"]["accepted_decisions"] == 1
    assert json.loads(report.read_text())["models"]["qwen"]["predictions"][0]["valid"] is True
    monkeypatch.setattr(modal_app, "infer_batch", SimpleNamespace(remote=lambda *a: []))
    with pytest.raises(RuntimeError, match="different number of decisions"):
        modal_app.main(mode="compare", validation_file=str(validation), models="qwen")
    monkeypatch.setattr(modal_app, "infer_batch", SimpleNamespace(remote=lambda *a: [{"text": json.dumps(decision)}]))
    with pytest.raises(ValueError, match="models must"):
        modal_app.main(mode="compare", validation_file=str(validation), models="qwen,qwen")
    with pytest.raises(ValueError, match="adapter-path comparison"):
        modal_app.main(mode="compare", validation_file=str(validation), adapter_path="/adapter")
    monkeypatch.setattr(modal_app, "infer_batch", SimpleNamespace(remote=lambda *a: [{"text": "invalid"}]))
    modal_app.main(mode="compare", validation_file=str(validation), models="qwen", limit=1)
    assert json.loads(capsys.readouterr().out)["models"]["qwen"]["valid_outputs"] == 0
    validation.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty validation"):
        modal_app.main(mode="compare", validation_file=str(validation))
