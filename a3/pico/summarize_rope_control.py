"""Keep the follow-up control separate from the original frozen pilot report."""
import hashlib
import json
import statistics
from pathlib import Path


def verify_additive_source_change():
    text = Path(__file__).with_name('variants.py').read_text()
    text = text.replace(", 'fox_rope')", ")")
    text = text.replace("variant in ('fox')", "variant == 'fox'")
    text = text.replace("('fox', 'fox_rope', 'dcmha')", "('fox', 'dcmha')")
    original = 'a5e9f85e3d63972b1bf0e1015208f048b557fb97f48bc71b0f8f38cadefe4b70'
    hashes = [hashlib.sha256(text.encode()).hexdigest(),
              hashlib.sha256(text.replace('\n', '\r\n').encode()).hexdigest()]
    assert original in hashes, 'Unexpected change beyond adding the RoPE control'


def summarize():
    verify_additive_source_change()
    folder = Path(__file__).resolve().parents[1] / 'pilot_results'
    old = json.loads((folder / 'sweep.json').read_text())
    new = json.loads((folder / 'rope-control.json').read_text())
    assert all(c['status'] == 'passed' for c in new['checks'])
    rows = [r for r in old if r['variant'] in ('baseline', 'fox')] + new['results']
    assert len(rows) == 6
    for variant in ('baseline', 'fox', 'fox_rope'):
        group = [r for r in rows if r['variant'] == variant]
        assert {r['seed'] for r in group} == {42, 43}
    assert all(r['status'] == 'complete' and r['tokens_seen'] == 16384000 for r in rows)
    for field in ('model', 'upstream_commit', 'precision', 'tokens_per_step', 'gpu'):
        assert len({json.dumps(r[field], sort_keys=True) for r in rows}) == 1, field
    lines = ['# FoX with RoPE: follow-up control', '',
             'Two seeds, 500 steps and 16.384M tokens each. Lower BPB is better.', '',
             '| Configuration | Seed 42 BPB | Seed 43 BPB | Mean BPB | Mean tokens/s |',
             '|---|---:|---:|---:|---:|']
    for variant in ('baseline', 'fox', 'fox_rope'):
        group = sorted((r for r in rows if r['variant'] == variant), key=lambda r: r['seed'])
        mean = statistics.mean(r['val_bpb'] for r in group)
        speed = statistics.mean(r['steady_tokens_per_second'] for r in group)
        lines.append(f"| {variant} | {group[0]['val_bpb']:.5f} | {group[1]['val_bpb']:.5f} | {mean:.5f} | {speed:,.0f} |")
    lines += ['', '## Interpretation limits', '',
              '- baseline: RoPE, no forgetting gates; fox: gates without RoPE; fox_rope: gates with RoPE.',
              '- New CPU/GPU controls verify identical initial FoX parameters and identity rotations recover original FoX outputs.',
              '- Source audit verifies old variant code is unchanged apart from adding the fox_rope branch.',
              '- Existing baseline/FoX measurements are historical, not concurrent timing controls.',
              '- The no-gates/no-RoPE cell remains untested: this is not the full factorial experiment.',
              '- Throughput comparisons include dense-reference versus optimized-attention kernel differences.',
              '- Two seeds do not establish statistical significance or larger-model performance.', '',
              '## New run provenance', '']
    for r in new['results']:
        lines.append(f"- [fox_rope seed {r['seed']}]({r['wandb_url']}), source SHA256 `{r['variant_source_sha256']}`.")
    seconds = sum(r['elapsed_seconds'] for r in new['results'])
    lines += ['', f'New successful seed-function time: {seconds:.1f}s; GPU-only estimate ${seconds * .001097:.2f}.',
              'This excludes tests, startup, CPU/RAM, and is not the final invoice.', '']
    report = '\n'.join(lines)
    (folder / 'ROPE_CONTROL.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    summarize()
