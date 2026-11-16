"""No-cloud Part 5 launcher contracts and distillation script syntax."""
import ast
from pathlib import Path
import pytest

from launcher_config import build_command, MODULES, OUTPUT_ROOT

HERE = Path(__file__).parent


def test_distillation_script_parses_without_importing_modal_or_loading_models():
    source = (HERE / "nanochat" / "scripts" / "chat_distill.py").read_text()
    ast.parse(source)
    assert "torch.no_grad()" in source
    assert "distillation_loss(" in source
    assert "teacher.eval()" in source
    assert '"distillation_config": user_config' in source


@pytest.mark.parametrize("mode,processes,steps", [
    ("dry", 1, "1"), ("smoke", 8, "1"), ("full", 8, None),
])
def test_distillation_launch_shape(mode, processes, steps):
    command = build_command(
        "distill", mode, "a4-p5-main", ["--num-epochs", "1"],
        start="a4-sft-default", teacher_tag="a4-p4-completion-v1"
    )
    assert MODULES["distill"] == "scripts.chat_distill"
    assert OUTPUT_ROOT["distill"] == "/vol/chatrl_checkpoints"
    assert "scripts.chat_distill" in command
    assert command[command.index("--model-tag") + 1] == "a4-sft-default"
    assert command[command.index("--model-step") + 1] == "466"
    assert command[command.index("--teacher-tag") + 1] == "a4-p4-completion-v1"
    assert command[command.index("--teacher-step") + 1] == "466"
    assert command[command.index("--output-tag") + 1] == "a4-p5-main"
    assert "--reward-mode" in command and command[command.index("--reward-mode") + 1] == "baseline"
    if steps:
        assert command[command.index("--max-steps") + 1] == steps
    else:
        assert "--max-steps" not in command
    if processes == 8:
        assert command[command.index("--nproc_per_node=8")] == "--nproc_per_node=8"
    else:
        assert "--nproc_per_node=8" not in command


@pytest.mark.parametrize("extra", [
    ["--teacher-tag", "other"], ["--teacher-step", "10"],
    ["--model-tag", "unsafe"], ["--output-tag", "reused"],
    ["--resume-from", "other"], ["--max-steps", "999"],
])
def test_smoke_rejects_experiment_overrides(extra):
    with pytest.raises(ValueError):
        build_command("distill", "smoke", "fresh", extra,
                      start="a4-sft-default", teacher_tag="a4-p4-completion-v1")


def test_teacher_required_and_reward_shaping_prohibited():
    with pytest.raises(ValueError):
        build_command("distill", "full", "fresh", [], start="a4-sft-default")
    with pytest.raises(ValueError):
        build_command("distill", "full", "fresh", [],
                      start="a4-sft-default", teacher_tag="a4-p4-completion-v1",
                      reward_mode="completion")


def test_full_distillation_resume_uses_new_output_and_original_checkpoint():
    command = build_command(
        "distill", "full", "a4-p5-distill-low-lr-resume-v1",
        ["--init-lr-frac", "0.01"],
        start="a4-sft-default", teacher_tag="a4-p4-completion-v1",
        resume_from="a4-p5-distill-low-lr-v1", resume_step=180,
    )
    assert command[command.index("--resume-from") + 1] == "a4-p5-distill-low-lr-v1"
    assert command[command.index("--resume-step") + 1] == "180"
    assert command[command.index("--output-tag") + 1] == "a4-p5-distill-low-lr-resume-v1"
    assert command[command.index("--init-lr-frac") + 1] == "0.01"


@pytest.mark.parametrize("source,step,mode,run", [
    ("", 180, "full", "new"),
    ("source", None, "full", "new"),
    ("source", 180, "smoke", "new"),
    ("source", 466, "full", "new"),
    ("source", -1, "full", "new"),
    ("source", 180, "full", "source"),
    ("../source", 180, "full", "new"),
])
def test_invalid_resume_configuration_rejected(source, step, mode, run):
    with pytest.raises(ValueError):
        build_command("distill", mode, run, [],
                      start="a4-sft-default", teacher_tag="a4-p4-completion-v1",
                      resume_from=source, resume_step=step)
