"""Run from ALIGN: a3/.venv/Scripts/modal run a3/pico_modal.py --stage smoke

Stages: prepare (CPU), smoke (GPU), sweep (three serial GPU calls).
No deployment, no retries, no automatic repeat runs. $40 approved pilot ceiling.
"""
from pathlib import Path
import json
import subprocess
import time
import sys
import os
import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App('a3-picochat-pilot')
CPU_ONLY = os.environ.get('A3_CPU_PREP_ONLY') == '1'
volume = modal.Volume.from_name('a3-picochat-pilot', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('git', 'build-essential')
         .pip_install('uv')
         .add_local_file(str(ROOT / 'nanochat' / 'pyproject.toml'), '/root/nanochat/pyproject.toml', copy=True)
         .add_local_file(str(ROOT / 'nanochat' / 'uv.lock'), '/root/nanochat/uv.lock', copy=True)
         .workdir('/root/nanochat')
         .run_commands('uv sync --frozen --extra gpu --no-install-project')
         .env({'PATH': '/root/nanochat/.venv/bin:/usr/local/bin:/usr/bin:/bin',
               'PYTHONPATH': '/root/nanochat:/root/a3', 'OMP_NUM_THREADS': '1',
               'NANOCHAT_BASE_DIR': '/vol/cache', 'HF_HOME': '/vol/hf',
               'PYTORCH_ALLOC_CONF': 'expandable_segments:True'})
         .add_local_dir(str(ROOT / 'nanochat' / 'nanochat'), '/root/nanochat/nanochat')
         .add_local_dir(str(ROOT / 'nanochat' / 'scripts'), '/root/nanochat/scripts')
         .add_local_dir(str(ROOT / 'a3' / 'pico'), '/root/a3/pico'))


def activate():
    # Modal's worker uses its own interpreter; subprocess PATH alone is insufficient.
    import site
    site.addsitedir('/root/nanochat/.venv/lib/python3.12/site-packages')


@app.function(image=image, volumes={'/vol': volume}, cpu=4, memory=16384,
              timeout=1800, retries=0, scaledown_window=2)
def prepare():
    activate()
    from pico.tests import run
    # CPU correctness tests must pass before allocating training hardware.
    tests = run('cpu')
    print(json.dumps(tests), flush=True)
    subprocess.run(['python', '-m', 'nanochat.dataset', '-n', '4'], check=True)
    if not Path('/vol/cache/tokenizer/tokenizer.pkl').exists():
        subprocess.run(['python', '-m', 'scripts.tok_train', '--max-chars=100000000',
                        '--vocab-size=32768'], check=True)
    volume.commit()
    return {'cpu_tests': tests, 'prepared': True}


@app.function(image=image, volumes={'/vol': volume}, gpu=None if CPU_ONLY else 'H100', cpu=4, memory=16384,
              timeout=600, retries=0, scaledown_window=2)
def smoke():
    activate()
    from pico.tests import run
    return run('cuda')


@app.function(image=image, volumes={'/vol': volume}, gpu=None if CPU_ONLY else 'H100', cpu=4, memory=16384,
              secrets=[modal.Secret.from_name('wandb-secret')],
              timeout=2400, retries=0, scaledown_window=2)
def experiment(variant: str):
    activate()
    from pico.variants import VARIANTS
    from pico.train import train
    assert variant in VARIANTS
    results = []
    for seed in (42, 43):
        path = f'/vol/results/pilot-v1/{variant}/seed{seed}'
        existing = Path(path) / 'result.json'
        if existing.exists():
            results.append(json.loads(existing.read_text()))
            continue
        results.append(train(variant, seed, path))
        volume.commit()
    return json.loads(json.dumps(results))


@app.function(image=modal.Image.debian_slim(python_version='3.12'),
              volumes={'/vol': volume}, timeout=60, cpu=.25, memory=256,
              retries=0, scaledown_window=2)
def collect():
    return [json.loads(p.read_text()) for p in Path('/vol/results/pilot-v1').glob('*/seed*/result.json')]


@app.local_entrypoint()
def main(stage: str = 'prepare', variant: str = 'baseline'):
    if CPU_ONLY and stage != 'prepare':
        raise ValueError('CPU-only mode permits preparation, never training.')
    dest = ROOT / 'a3' / 'pilot_results'
    dest.mkdir(exist_ok=True)
    if stage == 'prepare':
        result = prepare.remote()
    elif stage == 'collect':
        result = collect.remote()
        (dest / 'sweep.json').write_text(json.dumps(result, indent=2))
    elif stage == 'smoke':
        result = smoke.remote()
    elif stage == 'single':
        result = experiment.remote(variant)
        (dest / f'{variant}.json').write_text(json.dumps(result, indent=2))
    elif stage == 'variants':
        result = []
        for row in experiment.map(('cla2', 'fox'), order_outputs=False):
            result.extend(row)
            (dest / 'variants.json').write_text(json.dumps(result, indent=2))
    elif stage == 'sweep':
        # 3*2400s + 600s smoke = ~8.6 GPU dollars at $0.001097/s.
        # CPU/RAM/build allowance leaves substantial room below $40.
        # Do not rerun failed jobs automatically or silently enlarge these limits.
        result = []
        for variant in ('baseline', 'cla2', 'fox'):
            row = experiment.remote(variant)
            result.extend(row)
            (dest / 'sweep.json').write_text(json.dumps(result, indent=2))
    else:
        raise ValueError(stage)
    (dest / f'{stage}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
