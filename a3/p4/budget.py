"""Conservative launch reservations, separate from actual Modal billing."""
import json
from pathlib import Path
import time
import uuid

from .artifacts import append_jsonl
from .config import MAX_BUDGET_USD


def reserve(path, label, config, *, seconds, gpu=False):
    """Reserve a padded estimate; never automatically release failed calls.

    This is an additional guard, not an invoice API or replacement for the
    account's usage limit. CPU/memory bursts are capped in Modal declarations.
    """
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    rates = config["rates"]
    per_second = config["cpu"] * rates["cpu_core_per_second"] + config["memory_mb"] / 1024 * rates["memory_gib_per_second"]
    if gpu:
        per_second += rates["gpu_per_second"]
    amount = (seconds + 300) * per_second * 1.25
    used = sum(row["reserved_usd"] for row in rows)
    if used + amount > min(config["budget_usd"], MAX_BUDGET_USD):
        raise RuntimeError(f"Launch exceeds remaining budget reservation: ${used:.2f} + ${amount:.2f}")
    row = {"id": uuid.uuid4().hex, "timestamp": time.time(), "label": label,
           "reserved_usd": amount, "cumulative_reserved_usd": used + amount,
           "purpose": "conservative reservation, not actual billed usage", "gpu": gpu,
           "function_timeout_seconds": seconds, "startup_allowance_seconds": 300,
           "additional_multiplier": 1.25}
    append_jsonl(path, row)
    return row
