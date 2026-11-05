"""Native BPB and a small, fixed base-model question comparison. No CORE."""
import json
from pathlib import Path
import torch
from nanochat.common import compute_init
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from A3part5.fox import install


def measure(model, tokenizer, output):
    output = Path(output)
    model.eval()
    # A fixed evaluation microbatch allows the same packed stream for both sizes.
    rows = []
    for context in (512, 2048):
        model.config.sequence_len = context
        model.window_sizes = model._compute_window_sizes(model.config)
        loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, 8, context, "val")
        bpb = evaluate_bpb(model, loader, 262144 // (8 * context), get_token_bytes(device="cuda"))
        rows.append({"context": context, "nominal_positions": 262144, "bpb": bpb})
        (output / "bpb.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows[-1]), flush=True)
    model.config.sequence_len = 512
    model.window_sizes = model._compute_window_sizes(model.config)
    spec = json.loads(Path(__file__).with_name("evaluation_questions.json").read_text())
    answers = []
    with torch.inference_mode():
        for question in spec["questions"]:
            prefix = tokenizer(question["prompt"], prepend="<|bos|>")
            ids = torch.tensor([prefix], device="cuda", dtype=torch.long)
            for _ in range(spec["max_new_tokens"]):
                token = model(ids[:, -512:])[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat((ids, token), dim=1)
            answer = tokenizer.decode(ids[0, len(prefix):].tolist())
            answers.append(dict(question, completion=answer, grade=None))
            (output / "answers.json").write_text(json.dumps(answers, indent=2))
    return rows


def pico(output):
    """Reproduce the actual Part 2 recipe using the shared Part 5 tokenizer."""
    from pico.train import train
    from pico.variants import build
    output = Path(output)
    result = train("fox", 42, str(output), steps=500, seconds=1200)
    if result["completed_steps"] != 500:
        raise RuntimeError("Pico reference did not finish 500 steps")
    model = build("fox", seed=42)
    saved = torch.load(output / "model.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    rows = measure(model, get_tokenizer(), output)
    counts = model.num_scaling_params()
    (output / "comparison.json").write_text(json.dumps({"bpb": rows, "counts": counts}, indent=2))


def nano(output):
    install()
    compute_init("cuda")
    from nanochat.checkpoint_manager import build_model
    output = Path(output)
    folder = output / "base_checkpoints" / "fox-d24"
    step = max(int(p.stem.split("_")[-1]) for p in folder.glob("meta_*.json"))
    model, tokenizer, meta = build_model(str(folder), step, torch.device("cuda"), "eval")
    rows = measure(model, tokenizer, output)
    prediction = json.loads((output / "scaling_prediction.json").read_text())
    actual = rows[0]["bpb"]
    result = {"checkpoint_step": step, "bpb": rows, "predicted_bpb_512": prediction["predicted_bpb_512"],
              "prediction_error": actual - prediction["predicted_bpb_512"],
              "actual_training_tokens": step * meta["total_batch_size"],
              "answers_require_manual_review": True}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import sys
    (pico if sys.argv[1] == "pico" else nano)(sys.argv[2])
