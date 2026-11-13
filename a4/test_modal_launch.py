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


def test_latest_checkpoint_skips_a_step_that_was_cut_off_mid_save():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make(root, "run", [60, 120])
        (root / "run" / "meta_000120.json").unlink()  # model file written, job killed before the meta file
        assert launch.latest_checkpoint(root, ["run"]) == ("run", 60)
        (root / "run" / "meta_000060.json").write_text('{"model_con')  # meta file cut off
        assert launch.latest_checkpoint(root, ["run"]) is None


def test_latest_checkpoint_looks_in_every_folder_given():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make(root, "run", [60])
        make(root, "tag", [120])  # a job killed before the move leaves its newest steps under the tag
        assert launch.latest_checkpoint(root, ["run", "tag"]) == ("tag", 120)


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


def test_mark_run_tells_finished_from_unfinished():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        launch.mark_run(folder, "run", completed=False)
        assert (folder / "INCOMPLETE").exists() and not (folder / "COMPLETE").exists()
        launch.mark_run(folder, "run", completed=True)  # a resumed job that finishes replaces the earlier marker
        assert (folder / "COMPLETE").exists() and not (folder / "INCOMPLETE").exists()
