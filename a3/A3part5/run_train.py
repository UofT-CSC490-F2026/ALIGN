"""Invoke upstream base_train; add durable logging and checkpoint notifications."""
import json
import os
from pathlib import Path
import runpy
import torch.distributed as dist
import wandb
from A3part5.fox import install


def main():
    install()
    import nanochat.checkpoint_manager as checkpoints
    output = Path(os.environ["P5_OUTPUT"])
    native_save = checkpoints.save_checkpoint

    def save(*args, **kwargs):
        native_save(*args, **kwargs)
        if dist.is_initialized():
            dist.barrier()
        if int(os.environ.get("RANK", "0")) == 0:
            print("P5_CHECKPOINT_READY", flush=True)
    checkpoints.save_checkpoint = save

    # Preserve the native training logger; also persist its actual logged metrics.
    native_init = wandb.init
    def init(*args, **kwargs):
        kwargs["project"] = "a3-part5-fox"
        kwargs["dir"] = str(output)
        run = native_init(*args, **kwargs)
        (output / "wandb.json").write_text(json.dumps({"url": run.url}))
        print(f"P5_WANDB {run.url}", flush=True)
        native_log = run.log
        def log(data, *a, **kw):
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(data, default=str) + "\n")
            return native_log(data, *a, **kw)
        run.log = log
        return run
    wandb.init = init
    runpy.run_module("scripts.base_train", run_name="__main__")


if __name__ == "__main__":
    main()
