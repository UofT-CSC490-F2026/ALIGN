"""Summarize completed JSON results without inventing unavailable metrics."""
import json
import statistics
from pathlib import Path


def summarize(folder):
    folder = Path(folder)
    if (folder / 'baseline.json').exists() and (folder / 'variants.json').exists():
        rows = json.loads((folder / 'baseline.json').read_text()) + json.loads((folder / 'variants.json').read_text())
        (folder / 'sweep.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    else:
        rows = json.loads((folder / 'sweep.json').read_text())
    variants = ('baseline', 'cla2', 'fox')
    # Preserve the historical sweep on disk; summarize only the selected scope.
    rows = [r for r in rows if r['variant'] in variants]
    assert len({(r['variant'], r['seed']) for r in rows}) == len(rows), 'Duplicate results'
    assert len({r['variant_source_sha256'] for r in rows}) == 1, 'Architecture changed during sweep'
    assert len({json.dumps(r['model'], sort_keys=True) for r in rows}) == 1, 'Configuration mismatch'
    groups = {v: [r for r in rows if r['variant'] == v] for v in variants}
    baseline = groups['baseline']
    baseline_mean = statistics.mean(r['val_bpb'] for r in baseline) if baseline else None
    lines = ['# Picochat pilot results', '',
             'Short fixed-token screening; two seeds are not a significance test.', '',
             '| Variant | Status | BPB mean | BPB seeds | Delta vs baseline | Tokens/s | Peak GiB | Params |',
             '|---|---|---:|---|---:|---:|---:|---:|']
    for v in variants:
        group = groups[v]
        if not group:
            lines.append(f'| {v} | not run | - | - | - | - | - | - |')
            continue
        complete = all(r['status'] == 'complete' for r in group)
        matched = complete and len(group) == 2 and all(r['tokens_seen'] == 16384000 for r in group)
        mean = statistics.mean(r['val_bpb'] for r in group)
        delta = f'{mean - baseline_mean:+.5f}' if matched and baseline_mean is not None else 'not comparable'
        speeds = [r['steady_tokens_per_second'] for r in group if r['steady_tokens_per_second']]
        speed = f'{statistics.mean(speeds):,.0f}' if speeds else '-'
        seeds = ', '.join(f"{r['seed']}: {r['val_bpb']:.5f}" for r in group)
        lines.append(f"| {v} | {'complete' if matched else 'partial'} | {mean:.5f} | {seeds} | {delta} | {speed} | {max(r['peak_allocated_gib'] for r in group):.2f} | {group[0]['parameters']:,} |")
    seconds = sum(r['elapsed_seconds'] for r in rows)
    lines += ['', f'Recorded seed-function elapsed time: {seconds:,.1f} seconds.',
              f'GPU-only estimate for this recorded time: ${seconds * .001097:.2f}.',
              'This is NOT the final invoice: container startup, smoke tests, CPU/RAM,',
              'and any failed calls add cost. Credits are not deducted in this estimate.', '',
              '## Limitations', '',
              '- CLA2 is adjacent sharing with per-layer value embeddings, not FusedKV-Lite.',
              '- FoX uses dense reference attention; speed is implementation-specific.',
              '- No cached decoding, CORE, SFT, or final-model claims.',
              '- Two seeds, 16.384M tokens each, one short training schedule; do not extrapolate a winner.',
              '- GPU timings include implementation differences and hardware/run variability.',
              '- CUDA deterministic algorithms were not forced; exact reruns can differ slightly.',
              '', '## W&B runs', '']
    for r in rows:
        lines.append(f"- [{r['variant']} seed {r['seed']}]({r['wandb_url']})")
    if len(rows) == 6 and all(r['status'] == 'complete' for r in rows):
        means = {v: statistics.mean(r['val_bpb'] for r in groups[v]) for v in variants}
        best = min(means, key=means.get)
        lines += ['', '## Screening conclusion', '',
                  f'Lowest mean validation BPB: **{best}** ({means[best]:.5f}).',
                  'This nominates a follow-up experiment, not a universal architecture winner.',
                  'Retest promising changes at a longer training horizon and with more seeds.', '']
        for v in variants[1:]:
            pairs = []
            for seed in (42, 43):
                vr = next(r for r in groups[v] if r['seed'] == seed)
                br = next(r for r in baseline if r['seed'] == seed)
                pairs.append(vr['val_bpb'] - br['val_bpb'])
            lines.append(f'- {v}: paired BPB deltas {pairs[0]:+.5f}, {pairs[1]:+.5f}.')
    report = '\n'.join(lines) + '\n'
    (folder / 'REPORT.md').write_text(report, encoding='utf-8')
    return report


if __name__ == '__main__':
    import sys
    print(summarize(sys.argv[1]))
