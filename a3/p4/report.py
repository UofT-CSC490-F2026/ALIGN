"""Export Part 4 evidence tables and figures from saved JSONL files.

No missing result is imputed. Cost fields are resource-time estimates, never
represented as Modal invoices or credit deductions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import textwrap

from p4.evaluation import aggregate_results


COST_ESTIMATE_SCOPE = (
    "Training-chain worker entrypoint time only; excludes startup/image builds, "
    "CPU data preparation, rehearsal runs, storage, and final W&B finalization. "
    "This is not an invoice or a record of credit deductions."
)


def _read_jsonl(path):
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _number(row, names):
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return value
    return None


def _write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_table(rows):
    results = []
    attempts_by_stage, contexts_by_stage = {}, {}
    for row in rows:
        stage = row.get("stage", row.get("phase", ""))
        # Some final-evaluation rows omit attempt/context, but follow their
        # stage's identified worker in the same append-only log. Retain this
        # explicit inference separately from the original raw record.
        if row.get("attempt"):
            attempts_by_stage[stage] = row["attempt"]
        context = _number(row, ("context", "context_length"))
        if context is not None:
            contexts_by_stage[stage] = context
        attempt = row.get("attempt") or attempts_by_stage.get(stage, "")
        values = {
            "run_id": row.get("run_id", ""),
            "stage": stage,
            "attempt": attempt,
            "attempt_inferred": bool(attempt and not row.get("attempt")),
            "context_length": context if context is not None else contexts_by_stage.get(stage),
            "context_inferred": context is None and stage in contexts_by_stage,
            "step": _number(row, ("step", "global_step")),
            "tokens": _number(row, ("tokens", "cumulative_tokens", "tokens_seen")),
            "train_loss": _number(row, ("train/loss", "train_loss", "loss")),
            "val_bpb": _number(row, ("val/bpb", "validation_bpb", "val_bpb")),
            "tokens_per_second": _number(row, ("train/tokens_per_second", "tokens_per_second", "throughput")),
            "peak_allocated_gib": _number(row, ("peak_allocated_gib", "hardware/peak_allocated_gib")),
            "estimated_cost_usd": _number(row, ("estimated_cost_usd", "estimated_total_usd", "cost/estimated_usd")),
            "elapsed_seconds": _number(row, ("elapsed_seconds", "wall_seconds")),
            "event": row.get("event", ""),
            "raw_metric_json": json.dumps(row, sort_keys=True, allow_nan=False),
        }
        for name in row:
            match = re.fullmatch(r"val/bpb_context_([0-9]+)", name)
            if match:
                value = _number(row, (name,))
                if value is not None:
                    values["val_bpb_context_" + match.group(1)] = value
        if any(values[k] is not None for k in ("train_loss", "val_bpb", "tokens_per_second", "estimated_cost_usd", "peak_allocated_gib")):
            results.append(values)
        elif any(name.startswith("val_bpb_context_") for name in values):
            results.append(values)
    return results


def _series_label(row):
    labels = []
    if row.get("run_id"):
        labels.append(str(row["run_id"]))
    labels.append(str(row.get("stage") or "stage not recorded"))
    labels.append("attempt " + str(row.get("attempt") or "not recorded"))
    return " / ".join(labels)


def _metric_series(metrics, key):
    """Keep each stage/worker trajectory distinct and preserve log order.

    A resumed worker can start at a lower progress-token count than the failed
    worker reached. Joining/sorting both attempts would fabricate a trajectory.
    """
    series = {}
    for row in metrics:
        if row.get("tokens") is None or row.get(key) is None:
            continue
        series.setdefault(_series_label(row), []).append((row["tokens"], row[key]))
    return series


def _validation_series(metrics):
    """Prefer explicit evaluation-window fields over stage-dependent val/bpb."""
    series = {}
    for row in metrics:
        if row.get("tokens") is None:
            continue
        explicit = [(int(name.removeprefix("val_bpb_context_")), value)
                    for name, value in row.items() if name.startswith("val_bpb_context_")]
        if explicit:
            for context, value in sorted(explicit):
                label = _series_label(row) + f" / input window {context}"
                series.setdefault(label, []).append((row["tokens"], value))
        elif row.get("val_bpb") is not None:
            context = row.get("context_length")
            suffix = f"input window {context:g} (legacy metric)" if context is not None else "evaluation window not recorded"
            label = _series_label(row) + " / " + suffix
            series.setdefault(label, []).append((row["tokens"], row["val_bpb"]))
    return series


def _draw_figures(directory, metrics, aggregates):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [], "matplotlib_unavailable"
    saved = []
    plots = (
        ("train_loss", "Training loss (nats/token)", "training_loss.png"),
        ("val_bpb", "Validation bits per byte", "validation_bpb.png"),
        ("tokens_per_second", "Training tokens per second", "throughput.png"),
        ("peak_allocated_gib", "Peak allocated GPU memory (GiB)", "gpu_memory.png"),
        ("estimated_cost_usd", "Estimated chain worker cost (USD)", "estimated_cost.png"),
    )
    for key, label, filename in plots:
        series = _validation_series(metrics) if key == "val_bpb" else _metric_series(metrics, key)
        if not series:
            continue
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        for run_id, points in sorted(series.items()):
            axis.plot([p[0] / 1e6 for p in points], [p[1] for p in points], marker=".", label=run_id)
        axis.set_xlabel("Cumulative training tokens (millions)")
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        if key == "val_bpb":
            axis.set_title("Validation BPB by evaluation input window")
        # Even a single legacy series must disclose its stage/context labeling.
        axis.legend(fontsize=6, loc="best")
        fig.tight_layout()
        if key == "estimated_cost_usd":
            fig.subplots_adjust(bottom=.29)
            fig.text(.02, .02, textwrap.fill(COST_ESTIMATE_SCOPE, width=108), fontsize=6, va="bottom")
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)
        saved.append(filename)
    if aggregates:
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        labels = [f"{r['checkpoint']}\n{r['context_length']} tokens" for r in aggregates]
        axis.bar(range(len(aggregates)), [r["bpb"] for r in aggregates], color="#3568a8")
        axis.set_xticks(range(len(aggregates)), labels, fontsize=9)
        axis.set_ylabel("Held-out continuation bits per byte (lower is better)")
        axis.set_title("Recorded checkpoint/context cells")
        axis.grid(axis="y", alpha=.25)
        fig.tight_layout()
        filename = "context_comparison.png"
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)
        saved.append(filename)
    return saved, "generated" if saved else "no_plottable_metrics"


def create_report(run_dir, expected_checkpoints=("checkpoint1", "checkpoint2"), expected_contexts=(512, 2048)):
    """Create report/summary.json, CSV tables, README.md and available plots.

    Pass the directory of one experiment chain. Mixing independent seeds under
    one checkpoint label is rejected as duplicate examples; summarize each
    seed's chain separately before any explicitly paired multi-seed analysis.
    """
    from p4.artifacts import atomic_json

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    metric_rows, evaluation_rows, input_files = [], [], []
    for path in sorted(root.rglob("*.jsonl")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Evidence file escapes run directory: {path}")
        rows = _read_jsonl(path)
        relevant = False
        for row in rows:
            if row.get("event") == "evaluation_example":
                # Periodic validation uses separate labels and may have a
                # smaller example budget. It is not a final four-cell result.
                if row.get("checkpoint") in expected_checkpoints:
                    evaluation_rows.append(row)
                relevant = True
            elif any(name in row for name in ("train/loss", "train_loss", "val/bpb", "validation_bpb", "estimated_cost_usd", "estimated_total_usd", "tokens_per_second", "peak_allocated_gib")) or any(re.fullmatch(r"val/bpb_context_[0-9]+", name) for name in row):
                metric_rows.append(row)
                relevant = True
        if relevant:
            input_files.append(str(path.relative_to(root)))
    aggregates = aggregate_results(evaluation_rows)
    metrics = _metric_table(metric_rows)
    expected_cells = {(c, length) for c in expected_checkpoints for length in expected_contexts}
    observed_cells = {(row["checkpoint"], row["context_length"]) for row in aggregates}
    missing = sorted(expected_cells - observed_cells)
    example_sets = {}
    for row in evaluation_rows:
        key = row["checkpoint"], row["context_length"]
        example_sets.setdefault(key, set()).add(row["example_id"])
    matched = bool(example_sets) and len({frozenset(v) for v in example_sets.values()}) == 1
    status = "complete" if not missing and matched and expected_cells.issubset(observed_cells) else "incomplete"
    comparison_rows = []
    if len(expected_contexts) == 2:
        short, long = sorted(expected_contexts)
        by_cell = {(r["checkpoint"], r["context_length"]): r for r in aggregates}
        for checkpoint in expected_checkpoints:
            a, b = by_cell.get((checkpoint, short)), by_cell.get((checkpoint, long))
            same_examples = (a is not None and b is not None and
                             example_sets[(checkpoint, short)] == example_sets[(checkpoint, long)])
            comparison_rows.append({
                "checkpoint": checkpoint,
                "short_context": short,
                "long_context": long,
                "short_bpb": a["bpb"] if a else None,
                "long_bpb": b["bpb"] if b else None,
                "long_context_benefit_bpb": a["bpb"] - b["bpb"] if same_examples else None,
                "matched_examples": bool(same_examples),
            })
    directory = root / "report"
    directory.mkdir(exist_ok=True)
    _write_csv(directory / "evaluation_summary.csv", aggregates,
               ("checkpoint", "context_length", "examples", "scored_tokens", "target_bytes", "nll_nats", "bpb"))
    _write_csv(directory / "evaluation_examples.csv", evaluation_rows,
               ("checkpoint", "example_id", "context_length", "input_window_tokens", "target_suffix_tokens",
                "scored_tokens", "target_bytes", "nll_nats", "bpb", "target_sha256", "out_of_training_range"))
    _write_csv(directory / "context_benefit.csv", comparison_rows,
               ("checkpoint", "short_context", "long_context", "short_bpb", "long_bpb", "long_context_benefit_bpb", "matched_examples"))
    context_fields = sorted({name for row in metrics for name in row if name.startswith("val_bpb_context_")},
                            key=lambda name: int(name.removeprefix("val_bpb_context_")))
    _write_csv(directory / "training_metrics.csv", metrics,
               ("run_id", "stage", "attempt", "attempt_inferred", "context_length", "context_inferred",
                "step", "tokens", "train_loss", "val_bpb", *context_fields, "tokens_per_second",
                "peak_allocated_gib", "estimated_cost_usd", "elapsed_seconds", "event", "raw_metric_json"))
    figures, figure_status = _draw_figures(directory, metrics, aggregates)
    summary = {
        "schema_version": 1,
        "status": status,
        "missing_cells": [{"checkpoint": c, "context_length": length} for c, length in missing],
        "matched_example_sets": matched,
        "source_files": input_files,
        "evaluation": aggregates,
        "context_benefit": comparison_rows,
        "training_metric_rows": len(metrics),
        "figures": figures,
        "figure_status": figure_status,
        "cost_estimate_scope": COST_ESTIMATE_SCOPE,
        "plotting_policy": {
            "validation": "Separate explicit evaluation-input-window series; generic legacy metrics retain stage/context labels.",
            "attempts": "Separate worker attempts and stages, in saved log order; do not join rolled-back progress trajectories.",
            "inferred_metadata": "Rows missing attempt/context inherit the latest identified worker/context in that stage's saved log; inference is flagged in CSV.",
            "raw_metrics": "Original JSONL files are unchanged; raw_metric_json also preserves each selected source record in CSV.",
        },
        "limitations": [
            "Checkpoint 2 has additional training tokens; before/after differences do not isolate context length as the only cause.",
            "Checkpoint 1 evaluated at a larger input window is outside its training context range.",
            COST_ESTIMATE_SCOPE,
            "Matched target suffixes are scored; input window length includes preceding target tokens.",
        ],
    }
    atomic_json(directory / "summary.json", summary)
    lines = ["# Part 4 saved-evidence report", "", f"Evaluation status: **{status}**.", "",
             "| Checkpoint | Input window | Examples | BPB |", "|---|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {row['checkpoint']} | {row['context_length']} | {row['examples']} | {row['bpb']:.6f} |")
    if not aggregates:
        lines += ["", "No evaluation examples have been recorded."]
    if missing:
        lines += ["", "Missing cells: " + ", ".join(f"{c}/{length}" for c, length in missing) + "."]
    if evaluation_rows and not matched:
        lines += ["", "Example sets differ between cells; the four-cell comparison is incomplete."]
    lines += ["", "## Interpretation", ""] + ["- " + line for line in summary["limitations"]]
    lines += ["", "## Plotting policy", ""] + ["- " + line for line in summary["plotting_policy"].values()]
    lines += ["", "## Figures", ""]
    if figures:
        for filename in figures:
            lines.append(f"![{filename.removesuffix('.png').replace('_', ' ')}]({filename})")
            lines.append("")
    else:
        lines.append(f"No figures generated: {figure_status}.")
    lines += ["", "Full per-example token IDs/losses remain in the source JSONL files. CSV files are convenient summaries.", ""]
    (directory / "README.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(create_report(args.run_dir), indent=2))
