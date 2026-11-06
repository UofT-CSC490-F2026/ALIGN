"""Validate and summarize the independently persisted Part 3 runs."""
import json
import statistics as stats
from pathlib import Path


def summarize():
    folder = Path(__file__).resolve().parents[1] / 'hardware_results'
    workers = [json.loads((folder / f'{h}.json').read_text()) for h in ('A10', 'A100-80GB', 'H100')]
    rows = [r for w in workers for r in w['results']]
    assert len(rows) == 12
    assert len({(r['hardware'], r['variant'], r['seed']) for r in rows}) == 12
    assert all(r['status'] == 'complete' and r['tokens_seen'] == 16384000 and r['completed_steps'] == 500 for r in rows)
    for key in ('model', 'variant_source_sha256', 'upstream_commit', 'precision', 'steps', 'tokens_per_step'):
        assert len({json.dumps(r[key], sort_keys=True) for r in rows}) == 1, key
    # Preserve raw files. The delivered A10 device was A10G, whose vendor
    # datasheet specifies 70 dense BF16 TFLOP/s, not generic A10's 125.
    for r in rows:
        r['raw_logged_mfu_percent'] = r['estimated_mfu_percent']
        r['raw_logged_peak_flops'] = r['dense_bf16_peak_flops']
        if 'a10g' in r['gpu'].lower():
            r['dense_bf16_peak_flops'] = 70e12
            r['estimated_mfu_percent'] = 100 * r['model_flops_per_token'] * r['steady_tokens_per_second'] / 70e12
            r['mfu_correction'] = 'A10G 70 dense BF16 TFLOP/s: NVIDIA/AWS Feb 2022 datasheet; raw live logs used generic A10 125'
    (folder / 'reported_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    lines = ['# Part 3 hardware results', '',
             'Two seeds per configuration; throughput/MFU and times averaged across seeds.', '',
             '| GPU | Variant | Train s | First 5 steps s | Elapsed s | Tokens/s | MFU % | GPU $/run (elapsed) | BPB |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    grouped = []
    curves = []
    for hardware in ('A10', 'A100-80GB', 'H100'):
        for variant in ('baseline', 'fox'):
            group = [r for r in rows if r['hardware'] == hardware and r['variant'] == variant]
            mean = lambda key: stats.mean(r[key] for r in group)
            summary = dict(hardware=hardware, variant=variant,
                           **{key: mean(key) for key in ('train_seconds', 'warmup_step_seconds', 'elapsed_seconds', 'steady_tokens_per_second', 'estimated_mfu_percent', 'seed_elapsed_gpu_cost_estimate', 'val_bpb')})
            summary['seeds'] = [{k: r[k] for k in ('seed', 'val_bpb', 'steady_tokens_per_second', 'estimated_mfu_percent', 'wandb_url')} for r in group]
            summary['backends'] = sorted({r['attention_backend'] for r in group})
            grouped.append(summary)
            points = []
            for step in (100, 200, 300, 400, 500):
                cost = stats.mean(sum(r['step_seconds'][:step]) * r['gpu_dollars_per_second'] for r in group)
                bpb = stats.mean(next(p['val/bpb'] for p in r['history'] if p['step'] == step) for r in group)
                points.append((cost, bpb))
            curves.append(dict(hardware=hardware, variant=variant, points=points))
            lines.append(f"| {hardware} | {variant} | {mean('train_seconds'):.1f} | {mean('warmup_step_seconds'):.1f} | {mean('elapsed_seconds'):.1f} | {mean('steady_tokens_per_second'):,.0f} | {mean('estimated_mfu_percent'):.2f} | {mean('seed_elapsed_gpu_cost_estimate'):.3f} | {mean('val_bpb'):.5f} |")
    lines += ['', '## Provenance and per-seed measurements', '']
    for r in rows:
        lines.append(f"- [{r['hardware']} {r['variant']} seed {r['seed']}]({r['wandb_url']}): BPB {r['val_bpb']:.6f}, {r['steady_tokens_per_second']:,.0f} tokens/s, estimated MFU {r['estimated_mfu_percent']:.2f}%; GPU {r['gpu']}; backend `{r['attention_backend']}`; FLOPs/token {r['model_flops_per_token']:,}.")
    total = sum(w['worker_elapsed_seconds'] * w['results'][0]['gpu_dollars_per_second'] for w in workers)
    cpu_ram = sum(w['worker_elapsed_seconds'] * (4*.0000131+16*.00000222) for w in workers)
    lines += ['', f'Worker-timer GPU estimate: ${total:.2f}; requested CPU/RAM estimate: ${cpu_ram:.2f}.',
              'Excludes startup before worker timer, image build/storage and any other usage; invoice unverified.', '',
              'MFU is approximate useful-model FLOPs normalized to actual-device dense BF16 peaks (A10G 70/A100 312/H100 989 TFLOP/s), not GPU utilization.',
              'A10G report MFU is corrected from the generic A10 125 TFLOP/s used in raw live logs. Raw JSON is preserved; reported_results.json records both values.',
              'A10G source: https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf',
              'FoX reference attention contains FP32 work and redundant masked computation; backend differences confound pure hardware attribution.',
              'Steady rates exclude first five steps; training time includes them and data loading, but excludes evaluation.',
              'Two seeds/runs per configuration are not a robust timing study or statistical significance test.', '']
    (folder / 'summary.json').write_text(json.dumps(grouped, indent=2), encoding='utf-8')
    (folder / 'curves.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')
    (folder / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    summarize()
