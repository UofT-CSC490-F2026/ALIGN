"""Part 3: three GPU types, baseline/FoX, two seeds; no retries or deployments."""
import json
import time
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent.parent
# Build images locally only; Modal hydrates remote functions with this image.
if modal.is_local():
    from pico_modal import image
else:
    image = modal.Image.debian_slim(python_version='3.12')
volume = modal.Volume.from_name('a3-picochat-pilot')


def activate():
    import site
    site.addsitedir('/root/nanochat/.venv/lib/python3.12/site-packages')

app = modal.App('a3-hardware-efficiency')
options = dict(image=image, volumes={'/vol': volume}, cpu=4, memory=16384,
               secrets=[modal.Secret.from_name('wandb-secret')],
               timeout=2400, retries=0, scaledown_window=2, max_containers=1)


def worker(hardware):
    activate()
    from pico.tests import run
    from pico.train import train
    started = time.monotonic()
    checks = run('cuda', variants=('baseline', 'fox'))
    print(json.dumps({'hardware': hardware, 'checks': checks}), flush=True)
    rows = []
    # Counterbalance first/second position across seeds within each GPU container.
    for seed, variant in ((42, 'baseline'), (42, 'fox'), (43, 'fox'), (43, 'baseline')):
        path = Path(f'/vol/results/hardware-v1/{hardware}/{variant}/seed{seed}')
        existing = path / 'result.json'
        if existing.exists():
            row = json.loads(existing.read_text())
        else:
            row = train(variant, seed, path, seconds=500, hardware=hardware)
            volume.commit()
        rows.append(row)
        if row['status'] != 'complete':
            raise RuntimeError(f'Partial run: {hardware}/{variant}/{seed}; not retried')
    return {'hardware': hardware, 'checks': checks, 'results': rows,
            'worker_elapsed_seconds': time.monotonic() - started}


@app.function(gpu='A10', **options)
def a10():
    return worker('A10')


@app.function(gpu='A100-80GB', **options)
def a100():
    return worker('A100-80GB')


@app.function(gpu='H100', **options)
def h100():
    return worker('H100')


@app.local_entrypoint()
def main(stage: str = 'run'):
    dest = ROOT / 'a3' / 'hardware_results'
    dest.mkdir(exist_ok=True)
    if stage == 'correct-mfu':
        rows = json.loads((dest / 'reported_results.json').read_text())
        print(json.dumps(correct_mfu.remote(rows)), flush=True)
        return
    assert stage == 'run'
    calls = [('A10', a10.spawn()), ('A100-80GB', a100.spawn()), ('H100', h100.spawn())]
    rows = []
    for hardware, call in calls:
        result = call.get()
        (dest / f'{hardware}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        rows.extend(result['results'])
        (dest / 'sweep.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps({'collected': hardware, 'runs': len(rows)}), flush=True)


@app.function(image=image, cpu=.5, memory=1024, timeout=120, retries=0,
              secrets=[modal.Secret.from_name('wandb-secret')], scaledown_window=2)
def correct_mfu(rows):
    """Add corrected report metrics to W&B summary without rewriting raw history."""
    activate()
    import wandb
    api = wandb.Api()
    updated = []
    for row in rows:
        if 'mfu_correction' not in row:
            continue
        parts = row['wandb_url'].rstrip('/').split('/')
        run = api.run('/'.join((parts[-4], parts[-3], parts[-1])))
        run.summary['report/estimated_mfu_percent'] = row['estimated_mfu_percent']
        run.summary['report/dense_bf16_peak_flops'] = row['dense_bf16_peak_flops']
        run.summary['report/mfu_correction'] = row['mfu_correction']
        run.summary.update()
        updated.append(row['wandb_url'])
    return {'updated_summaries': updated, 'raw_history_preserved': True}
