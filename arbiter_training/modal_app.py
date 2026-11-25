"""Modal inference for two base models and provenance-tracked Qwen LoRA training.

Upload prepared train.jsonl and validation.jsonl to the align-arbiter-data volume
before invoking train. Training writes a versioned adapter to a separate volume.
"""

import json
import hashlib
import time
from collections import Counter
from pathlib import Path

import modal

app = modal.App("align-arbiter-training")
data_volume = modal.Volume.from_name("align-arbiter-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("align-arbiter-checkpoints", create_if_missing=True)
cache_volume = modal.Volume.from_name("align-arbiter-model-cache", create_if_missing=True)

inference_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm>=0.11", "transformers>=5", "huggingface_hub"
)
training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "unsloth", "trl", "datasets", "transformers>=5"
)

MODELS = {"qwen": "Qwen/Qwen3.5-9B", "gpt-oss": "openai/gpt-oss-20b"}
MODEL_REVISIONS = {"qwen": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                   "gpt-oss": "6cee5e81ee83917806bbde320786a8fb61efebee"}


def _load_inference_engine(model_name: str, adapter_path: str):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if adapter_path and model_name != "qwen":
        raise ValueError("Only the Qwen adapter is supported by this training job")
    from vllm import LLM, SamplingParams

    model_id = MODELS[model_name]
    kwargs = {"model": model_id, "revision": MODEL_REVISIONS[model_name],
              "max_model_len": 8192, "enforce_eager": True}
    if adapter_path:
        from vllm.lora.request import LoRARequest
        kwargs.update(enable_lora=True, max_lora_rank=16)
        lora_request = LoRARequest("align-arbiter", 1, adapter_path)
    else:
        lora_request = None
    return LLM(**kwargs), lora_request


def _generate_batch(llm, lora_request, model_name, adapter_path, batch):
    from vllm import SamplingParams

    model_id = MODELS[model_name]
    tokenizer = llm.get_tokenizer()
    template_args = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "qwen":
        template_args["enable_thinking"] = False
    prompts = [tokenizer.apply_chat_template(messages, **template_args) for messages in batch]
    if model_name == "gpt-oss":
        # The Harmony template otherwise starts an analysis channel. Force the
        # final channel so the decision contract receives only the answer.
        prompts = [prompt + "<|channel|>final<|message|>" for prompt in prompts]
    params = SamplingParams(temperature=0, max_tokens=256)
    results = llm.generate(prompts, params, lora_request=lora_request)
    return [{"model": model_id, "model_revision": MODEL_REVISIONS[model_name],
             "adapter": adapter_path or None,
             "text": result.outputs[0].text,
             "finish_reason": result.outputs[0].finish_reason} for result in results]


@app.function(
    image=inference_image,
    gpu="H100",
    timeout=3600,
    volumes={"/root/.cache/huggingface": cache_volume,
             "/checkpoints": checkpoint_volume},
)
def infer_batch(model_name: str, batch: list[list[dict]], adapter_path: str = "") -> list[dict]:
    """Run a batch with one loaded base model or trained adapter."""
    llm, lora_request = _load_inference_engine(model_name, adapter_path)
    return _generate_batch(llm, lora_request, model_name, adapter_path, batch)


@app.cls(
    image=inference_image,
    gpu="H100",
    timeout=3600,
    volumes={"/root/.cache/huggingface": cache_volume,
             "/checkpoints": checkpoint_volume},
)
class ArbiterModel:
    """Keep a chosen base model or adapter loaded across harness decisions."""

    model_name: str = modal.parameter()
    adapter_path: str = modal.parameter(default="")

    @modal.enter()
    def setup(self):
        self.llm, self.lora_request = _load_inference_engine(
            self.model_name, self.adapter_path)

    @modal.method()
    def decide_batch(self, batch: list[list[dict]]) -> list[dict]:
        return _generate_batch(self.llm, self.lora_request, self.model_name,
                               self.adapter_path, batch)


def _records(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _decision_text(model_name, output):
    if model_name == "gpt-oss" and "assistantfinal" in output:
        return output.rsplit("assistantfinal", 1)[1].removeprefix(" json").strip()
    return output


@app.function(
    image=training_image,
    gpu="H100",
    timeout=21600,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume,
             "/root/.cache/huggingface": cache_volume},
)
def train_qwen(run_name: str, max_steps: int = 100,
               train_volume_path: str = "/data/train.jsonl",
               validation_volume_path: str = "/data/validation.jsonl") -> dict:
    """BF16 LoRA SFT using labeled, task-separated A2 checkpoints."""
    if not run_name.isascii() or not run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_name must contain only ASCII letters, digits, - or _")
    if not 1 <= max_steps <= 5000:
        raise ValueError("max_steps must be 1..5000")
    train_path, val_path = train_volume_path, validation_volume_path
    if not all(Path(path).is_relative_to("/data") for path in (train_path, val_path)):
        raise ValueError("Training input paths must be inside /data")
    train_rows, val_rows = list(_records(train_path)), list(_records(val_path))
    if len(train_rows) < 20 or len(val_rows) < 5:
        raise ValueError("Need at least 20 training and 5 validation checkpoints")
    train_actions = Counter(r["primary_action"] for r in train_rows)
    if sum(count for action, count in train_actions.items() if action != "continue") < 5:
        raise ValueError("Need at least 5 intervention examples before training")
    train_tasks = {r["task_key"] for r in train_rows}
    val_tasks = {r["task_key"] for r in val_rows}
    if train_tasks & val_tasks:
        raise ValueError("Train and validation tasks overlap")
    def repository(row):
        return (row["task_key"].split(":")[1] if ":" in row["task_key"]
                else row["task_key"])
    if {repository(r) for r in train_rows} & {repository(r) for r in val_rows}:
        raise ValueError("Train and validation repositories overlap")
    if any(r.get("label_source") not in {"human", "teacher_gpt"}
           for r in train_rows + val_rows):
        raise ValueError("Missing label provenance")
    output = Path("/checkpoints") / run_name
    if output.exists():
        raise ValueError("Checkpoint run name already exists")

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    max_length = 8192
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODELS["qwen"], max_seq_length=max_length,
        revision=MODEL_REVISIONS["qwen"],
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
        max_seq_length=max_length,
    )
    def as_dataset(rows):
        return Dataset.from_list([{
            "prompt": tokenizer.apply_chat_template(
                r["prompt"], tokenize=False, add_generation_prompt=True,
                enable_thinking=False),
            "completion": r["completion"][0]["content"] + tokenizer.eos_token,
        } for r in rows])
    interval = max(1, max_steps // 4)
    config = SFTConfig(
        output_dir=str(output / "training"), max_length=max_length,
        completion_only_loss=True, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, warmup_steps=min(10, max_steps // 10),
        max_steps=max_steps, logging_steps=interval, eval_strategy="steps",
        eval_steps=interval, save_strategy="steps", save_steps=interval,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, bf16=True, seed=3407, report_to=[],
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, args=config,
        train_dataset=as_dataset(train_rows), eval_dataset=as_dataset(val_rows),
    )
    trainer.train()
    adapter = output / "adapter"
    trainer.model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    metadata = {"model": MODELS["qwen"], "model_revision": MODEL_REVISIONS["qwen"],
                "run_name": run_name,
                "train_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
                "validation_sha256": hashlib.sha256(Path(val_path).read_bytes()).hexdigest(),
                "train_examples": len(train_rows), "validation_examples": len(val_rows),
                "train_tasks": len(train_tasks), "validation_tasks": len(val_tasks),
                "max_steps": max_steps, "seed": 3407,
                "label_sources": sorted({r["label_source"] for r in train_rows}),
                "train_actions": dict(train_actions),
                "synthetic_train_examples": sum(r.get("scenario_source") == "synthetic_counterfactual"
                                                for r in train_rows),
                "validation_review_status": sorted({r["review_status"] for r in val_rows}),
                "best_metric": trainer.state.best_metric,
                "adapter_path": str(adapter)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    checkpoint_volume.commit()
    return metadata


@app.local_entrypoint()
def main(mode: str = "infer", model: str = "qwen", checkpoint_file: str = "",
         adapter_path: str = "", run_name: str = "", steps: int = 100,
         validation_file: str = "", limit: int = 0, output_file: str = "",
         models: str = "qwen,gpt-oss",
         train_volume_path: str = "/data/train.jsonl",
         validation_volume_path: str = "/data/validation.jsonl"):
    if mode == "train":
        if not run_name:
            raise ValueError("--run-name is required for training")
        print(json.dumps(train_qwen.remote(run_name, steps, train_volume_path,
                                           validation_volume_path), indent=2))
        return
    from arbiter_training.contract import (messages_for, messages_for_observation,
                                           parse_decision)

    if mode == "infer":
        if not checkpoint_file:
            raise ValueError("--checkpoint-file is required for inference")
        observation = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
        messages = (messages_for(observation) if "task" in observation
                    else messages_for_observation(observation))
        response = infer_batch.remote(model, [messages], adapter_path)[0]
        try:
            response["decision"] = parse_decision(_decision_text(model, response["text"]))
        except ValueError as exc:
            response["parse_error"] = str(exc)
        print(json.dumps(response, indent=2))
        return
    if mode == "class-smoke":
        if not validation_file:
            raise ValueError("--validation-file is required for class-smoke")
        from arbiter_training.inference import ModalArbiter

        row = next(_records(validation_file))
        observation = json.loads(row["prompt"][1]["content"])
        remote_model = ArbiterModel(model_name=model, adapter_path=adapter_path)
        decision = ModalArbiter(model=model, adapter_path=adapter_path,
                                invalid_policy="raise",
                                remote_model=remote_model).decide(observation)
        print(json.dumps({"observation_task_key": observation["task_key"],
                          "decision": decision}, indent=2))
        return
    if mode != "compare" or not validation_file:
        raise ValueError("Use --mode infer, compare, train, or class-smoke")
    rows = list(_records(validation_file))
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty validation set")
    report = {"examples": len(rows),
              "validation_sha256": hashlib.sha256(Path(validation_file).read_bytes()).hexdigest(),
              "models": {}}
    candidates = [name.strip() for name in models.split(",") if name.strip()]
    if not candidates or len(set(candidates)) != len(candidates) or any(
        name not in MODELS for name in candidates
    ):
        raise ValueError("--models must be a comma-separated subset of qwen,gpt-oss")
    if adapter_path and candidates != ["qwen"]:
        raise ValueError("--adapter-path comparison requires --models qwen")
    for candidate in candidates:
        started = time.monotonic()
        outputs = infer_batch.remote(candidate, [r["prompt"] for r in rows],
                                     adapter_path if candidate == "qwen" else "")
        judged = []
        for row, result in zip(rows, outputs):
            try:
                decision = parse_decision(_decision_text(candidate, result["text"]))
                action = decision["action"]
                valid = True
            except ValueError:
                action, valid = None, False
            judged.append({"task_key": row["task_key"],
                           "checkpoint_event_uuid": row["checkpoint_event_uuid"],
                           "scenario_source": row.get("scenario_source", "public_trace"),
                           "gold_action": row["primary_action"],
                           "expected_actions": row["expected_actions"],
                           "predicted_action": action, "valid": valid,
                           "accepted": action in row["expected_actions"],
                           "raw_output": result["text"]})
        groups = {}
        for source in sorted({item["scenario_source"] for item in judged}):
            subset = [item for item in judged if item["scenario_source"] == source]
            groups[source] = {"examples": len(subset),
                              "accepted": sum(item["accepted"] for item in subset),
                              "valid": sum(item["valid"] for item in subset)}
        by_action = {}
        for action in sorted({item["gold_action"] for item in judged}):
            subset = [item for item in judged if item["gold_action"] == action]
            by_action[action] = {"examples": len(subset),
                                 "accepted": sum(item["accepted"] for item in subset)}
        report["models"][candidate + ("-adapter" if adapter_path else "")] = {
            "adapter_path": adapter_path or None,
            "model_revision": MODEL_REVISIONS[candidate],
            "valid_outputs": sum(item["valid"] for item in judged),
            "accepted_decisions": sum(item["accepted"] for item in judged),
            "cold_start_seconds": round(time.monotonic() - started, 2),
            "by_source": groups, "by_gold_action": by_action,
            "predictions": judged,
        }
    if output_file:
        Path(output_file).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(rows), "models": {
        name: {key: value for key, value in result.items() if key != "predictions"}
        for name, result in report["models"].items()}}, indent=2))
