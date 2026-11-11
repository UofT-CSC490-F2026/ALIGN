"""Unit tests for the checkpoint helpers in modal_launch.py (run with the modal tool's python: they need `import modal`)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import modal_launch as launch


def make(root, folder, steps):
    for step in steps:
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / f"model_{step:06d}.pt").write_text("x")
        (root / folder / f"meta_{step:06d}.json").write_text("{}")


def test_latest_checkpoint_picks_highest_step_in_named_folder():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make(root, "run", [60, 120, 90])
        make(root, "other", [300])
        assert launch.latest_checkpoint(root, ["run"]) == ("run", 120)
        assert launch.latest_checkpoint(root, ["missing"]) is None


def test_merge_move_adds_files_to_an_existing_folder():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make(root, "run", [60])
        make(root, "tag", [120])
        launch.merge_move(root / "tag", root / "run")
        assert not (root / "tag").exists()
        assert sorted(p.name for p in (root / "run").glob("model_*.pt")) == ["model_000060.pt", "model_000120.pt"]


def test_merge_move_creates_the_target():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make(root, "tag", [60])
        launch.merge_move(root / "tag", root / "run")
        assert (root / "run" / "model_000060.pt").exists() and not (root / "tag").exists()
