"""Resolve only the checked-in Part 4 code and pinned external dependency."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

P4_ROOT = Path(__file__).resolve().parent
A3_ROOT = P4_ROOT.parent
UPSTREAM = P4_ROOT / "vendor" / "nanochat"


def verify_jinhao_sources():
    manifest = json.loads((P4_ROOT / "JINHAO_SOURCE_MANIFEST.json").read_text())
    repository = A3_ROOT.parent
    for row in manifest["files"]:
        path = repository / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Jinhao source changed: {row['path']}")
    return {"checked_files": len(manifest["files"]), "unchanged": True}


def activate_local():
    from .config import UPSTREAM_COMMIT
    actual = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError("External nanochat checkout differs from the pinned commit")
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("Pinned nanochat has modified tracked files")
    for path in (A3_ROOT, UPSTREAM):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return verify_jinhao_sources()


if __name__ == "__main__":
    print(json.dumps(activate_local(), indent=2))
