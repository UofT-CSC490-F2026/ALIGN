"""Local-only tests for isolated RL/distillation recovery policy."""
import json
import pytest
from launcher_config import validate_resume_request, plan_recovery


def saved(root, run, step, valid=True):
    path = root / run
    path.mkdir(exist_ok=True)
    (path / f"model_{step:06d}.pt").write_text("weights")
    (path / f"meta_{step:06d}.json").write_text(
        json.dumps({"model_config": {"n_layer": 24}}) if valid else "not-json"
    )
    return path


def test_fresh_run(tmp_path):
    command = ["train", "--run", "new"]
    assert plan_recovery(tmp_path, tmp_path / "rollouts", "new", command) == (command, False)


def test_manual_resume_preserves_source(tmp_path):
    saved(tmp_path, "old", 180)
    command = ["train", "--resume-from", "old", "--resume-step", "180"]
    output, resumed = plan_recovery(tmp_path, tmp_path / "rollouts", "new", command, "old", 180)
    assert resumed and output == command
    assert (tmp_path / "old" / "model_000180.pt").exists()


def test_manual_resume_needs_fresh_target(tmp_path):
    saved(tmp_path, "old", 180)
    (tmp_path / "new").mkdir()
    with pytest.raises(FileExistsError):
        plan_recovery(tmp_path, tmp_path / "rollouts", "new", ["train"], "old", 180)


def test_invalid_resume_checkpoint_rejected(tmp_path):
    saved(tmp_path, "old", 180, valid=False)
    with pytest.raises(ValueError):
        plan_recovery(tmp_path, tmp_path / "rollouts", "new", ["train"], "old", 180)


def test_automatic_restart_uses_complete_checkpoint(tmp_path):
    command = ["train", "--run", "x"]
    path = saved(tmp_path, "x", 60)
    (path / "INCOMPLETE").write_text("x")
    (path / "launch_command.json").write_text(json.dumps(command))
    output, resumed = plan_recovery(tmp_path, tmp_path / "rollouts", "x", command)
    assert resumed and output[-4:] == ["--resume-from", "x", "--resume-step", "60"]


def test_automatic_restart_skips_invalid_newer_checkpoint(tmp_path):
    command = ["train"]
    path = saved(tmp_path, "x", 60)
    saved(tmp_path, "x", 120, valid=False)
    (path / "INCOMPLETE").write_text("x")
    (path / "launch_command.json").write_text(json.dumps(command))
    output, _ = plan_recovery(tmp_path, tmp_path / "rollouts", "x", command)
    assert output[-1] == "60"


def test_changed_configuration_rejected(tmp_path):
    path = saved(tmp_path, "x", 60)
    (path / "INCOMPLETE").write_text("x")
    (path / "launch_command.json").write_text(json.dumps(["train", "--lr", "0.01"]))
    with pytest.raises(ValueError):
        plan_recovery(tmp_path, tmp_path / "rollouts", "x", ["train", "--lr", "0.02"])


def test_completed_run_rejected(tmp_path):
    path = saved(tmp_path, "x", 60)
    (path / "COMPLETE").write_text("x")
    with pytest.raises(FileExistsError):
        plan_recovery(tmp_path, tmp_path / "rollouts", "x", ["train"])


def test_restart_of_manually_resumed_run(tmp_path):
    command = ["train", "--resume-from", "old", "--resume-step", "180"]
    path = saved(tmp_path, "new", 240)
    (path / "INCOMPLETE").write_text("new")
    (path / "launch_command.json").write_text(json.dumps(command))
    output, resumed = plan_recovery(tmp_path, tmp_path / "rollouts", "new", command, "old", 180)
    assert resumed and output[-4:] == ["--resume-from", "new", "--resume-step", "240"]


@pytest.mark.parametrize("script,mode,source,step", [
    ("rl", "smoke", "old", 60), ("distill", "dry", "old", 60),
    ("sft", "full", "old", 60), ("rl", "full", "", 60),
    ("rl", "full", "old", None), ("rl", "full", "../bad", 60),
    ("rl", "full", "new", 60), ("rl", "full", "old", -1),
])
def test_invalid_resume_request(script, mode, source, step):
    with pytest.raises(ValueError):
        validate_resume_request(script, mode, "new", source, step)


def test_no_resume_request():
    validate_resume_request("rl", "full", "new", "", None)
