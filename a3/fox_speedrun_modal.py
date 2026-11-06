"""Isolated 170-shard, 8-H100 FoX speedrun. No automatic retries."""
import json
from pathlib import Path
import modal

if modal.is_local():
    from pico_modal import image
else:
    image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('a3-fox-speedrun-170')
volume=modal.Volume.from_name('a3-picochat-pilot')
BASE='/vol/fox_speedrun_cache'
OUT='/vol/results/fox-speedrun-v1'


def activate():
    import os,site
    os.environ['NANOCHAT_BASE_DIR']=BASE
    site.addsitedir('/root/nanochat/.venv/lib/python3.12/site-packages')


@app.function(image=image,volumes={'/vol':volume},cpu=8,memory=32768,
              timeout=3600,retries=0,scaledown_window=2)
def prepare():
    activate()
    import subprocess,hashlib
    data=Path(BASE)/'base_data_climbmix'
    data.mkdir(parents=True,exist_ok=True)
    # Reuse existing shards read-only; download the additional shards separately.
    for i in (*range(24),6542):
        src=Path('/vol/final_cache/base_data_climbmix')/f'shard_{i:05d}.parquet'
        if src.exists() and not (data/src.name).exists():
            (data/src.name).symlink_to(src)
    subprocess.run(['python','-m','nanochat.dataset','-n','170','-w','8'],check=True,timeout=3000)
    files=sorted(p.name for p in data.glob('*.parquet'))
    assert files==[f'shard_{i:05d}.parquet' for i in (*range(170),6542)]
    # Identical upstream tokenizer recipe already completed in July-aligned preparation.
    import shutil
    tokenizer=Path(BASE)/'tokenizer'
    if not tokenizer.exists():
        shutil.copytree('/vol/july_aligned_cache/tokenizer',tokenizer)
    old=json.loads(Path('/vol/results/part2-july-aligned-v1/prepare.json').read_text())
    digest=hashlib.sha256((tokenizer/'tokenizer.pkl').read_bytes()).hexdigest()
    assert digest==old['tokenizer_sha256'] and old['tokenizer_chars']==2000000000
    from scripts.base_eval import EVAL_BUNDLE_URL,place_eval_bundle
    from nanochat.common import download_file_with_lock
    if not Path(BASE,'eval_bundle').exists():
        download_file_with_lock(EVAL_BUNDLE_URL,'eval_bundle.zip',postprocess_fn=place_eval_bundle)
    from pico.fox_speedrun import FoxGPT
    from nanochat.gpt import GPTConfig
    import torch,math
    def size(depth):
        with torch.device('meta'):
            model=FoxGPT(GPTConfig(n_layer=depth,n_embd=depth*64,n_head=depth//2,n_kv_head=depth//2))
        p=model.num_scaling_params()
        return p,p['transformer_matrices']+p['lm_head']
    p,n=size(24)
    _,ref=size(12)
    batch=2**round(math.log2(2**19*(n/ref)**.383))
    result=dict(parameters=p,scaling_parameters=n,global_batch=batch,steps=8*n//batch,
                tokens=(8*n//batch)*batch,tokenizer_sha256=digest,train_shards=170,
                stopping_rule='upstream ratio 8; not one full epoch',seed=42)
    Path(OUT).mkdir(parents=True,exist_ok=True)
    Path(OUT,'prepare.json').write_text(json.dumps(result,indent=2))
    volume.commit()
    return result


@app.function(image=image,volumes={'/vol':volume},gpu='H100',cpu=4,memory=16384,
              timeout=600,retries=0,scaledown_window=2)
def kernel_check():
    activate()
    from pico.test_fox_speedrun import run
    result=run()
    import hashlib
    result['model_source_sha256']=hashlib.sha256(Path('/root/a3/pico/fox_speedrun.py').read_bytes()).hexdigest()
    Path(OUT).mkdir(parents=True,exist_ok=True)
    Path(OUT,'kernel-check.json').write_text(json.dumps(result,indent=2))
    volume.commit()
    return result


@app.function(image=image,volumes={'/vol':volume},cpu=4,memory=8192,
              timeout=600,retries=0,scaledown_window=2)
def prepare_eval():
    activate()
    import torch,hashlib
    from nanochat.tokenizer import get_tokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    dest=Path(BASE,'fixed_eval.pt')
    if not dest.exists():
        loader=tokenizing_distributed_data_loader_bos_bestfit(get_tokenizer(),16,2048,split='val',device='cpu')
        batches=[]
        for _ in range(32):
            x,y=next(loader)
            batches.append((x.clone(),y.clone()))
        torch.save(batches,dest)
    result=dict(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                calibration_targets=524288,test_targets=524288,
                source='held-out shard 06542; fixed BOS-bestfit rows; first/second 256 rows')
    Path(OUT,'fixed-eval.json').write_text(json.dumps(result,indent=2))
    volume.commit()
    return result


def run_training(name,depth,gpus,smoke=False,seconds=13000):
    activate()
    import subprocess,os,hashlib,time
    check=json.loads(Path(OUT,'kernel-check.json').read_text())
    assert check['status']=='passed'
    assert check['model_source_sha256']==hashlib.sha256(Path('/root/a3/pico/fox_speedrun.py').read_bytes()).hexdigest()
    dest=Path(OUT,name)
    if dest.exists():
        raise RuntimeError(f'Paid attempt already exists: {name}')
    dest.mkdir(parents=True)
    (dest/'started.json').write_text(json.dumps({'time':time.time(),'gpus':gpus,'timeout':seconds}))
    volume.commit()
    env={**os.environ,'A3_RUN_DIR':str(dest),'TORCHINDUCTOR_COMPILE_THREADS':'2',
         'PYTHONUNBUFFERED':'1'}
    cmd=['torchrun','--standalone',f'--nproc_per_node={gpus}','-m','pico.speedrun_train',
         f'--depth={depth}','--target-param-data-ratio=8','--device-batch-size=16','--fp8',
         f'--run={name}',f'--model-tag={name}','--save-every=250']
    if smoke:
        cmd+=['--num-iterations=20','--eval-tokens=262144','--eval-every=20',
              '--core-metric-every=-1','--sample-every=-1']
    elif depth!=24:
        # Scaling calibration measures upstream BPB, not capabilities of these small models.
        cmd+=['--core-metric-every=-1','--sample-every=-1']
    try:
        subprocess.run(cmd,env=env,check=True,timeout=seconds)
    except BaseException as exc:
        (dest/'failure.json').write_text(json.dumps({'time':time.time(),'type':type(exc).__name__,'message':str(exc)}))
        raise
    finally:
        volume.commit()
    return json.loads((dest/'result.json').read_text())


@app.function(image=image,volumes={'/vol':volume},gpu='H100:8',cpu=16,memory=131072,
              timeout=900,retries=0,scaledown_window=2,max_containers=1,
              secrets=[modal.Secret.from_name('wandb-secret')])
def distributed_check():
    return run_training('distributed-smoke',24,8,smoke=True,seconds=840)


@app.function(image=image,volumes={'/vol':volume},gpu='H100!:8',cpu=(16,16),memory=(131072,131072),
              timeout=900,retries=0,scaledown_window=2,max_containers=1,
              secrets=[modal.Secret.from_name('wandb-secret')])
def h100_check():
    return run_training('distributed-smoke-h100',24,8,smoke=True,seconds=840)


@app.function(image=image,volumes={'/vol':volume},gpu='H100',cpu=4,memory=16384,
              timeout=4800,retries=0,scaledown_window=2,max_containers=1,
              secrets=[modal.Secret.from_name('wandb-secret')])
def calibrate():
    import time
    started=time.monotonic()
    results=[]
    for depth in (4,6,8):
        seconds=min(1800,int(4600-(time.monotonic()-started)))
        if seconds<300:
            raise RuntimeError('Calibration budget exhausted; no automatic retry')
        results.append(run_training(f'calibration-d{depth}',depth,1,seconds=seconds))
    return results


@app.function(image=image,volumes={'/vol':volume},gpu='H100!',cpu=(4,4),memory=(16384,16384),
              timeout=1800,retries=0,scaledown_window=2,max_containers=1,
              secrets=[modal.Secret.from_name('wandb-secret')])
def recover_calibration():
    """One explicitly authorized clean replacement; retain canceled run unchanged."""
    return run_training('calibration-d8-recovery',8,1,seconds=1740)


@app.function(image=image,volumes={'/vol':volume},cpu=1,memory=2048,
              timeout=180,retries=0,scaledown_window=2)
def predict():
    activate()
    import numpy as np,time,hashlib
    if Path(OUT,'prediction.json').exists() or Path(OUT,'final-fox-d24').exists():
        raise RuntimeError('Prediction must be frozen once before final training')
    names=['calibration-d4','calibration-d6','calibration-d8-recovery']
    rows=[json.loads(Path(OUT,name,'result.json').read_text()) for name in names]
    assert all(r['status']=='complete' for r in rows)
    # One point per independently trained scale; two parameters, no floor term.
    # Data grows with scaling parameter count, so this is a joint compute-optimal trajectory,
    # NOT separate identifiable model-size and data-size exponents.
    x=np.log([r['scaling_parameters'] for r in rows])
    assert len({r['fixed_eval_sha256'] for r in rows})==1
    y=np.log([r['calibration_bpb'] for r in rows])
    slope,intercept=np.polyfit(x,y,1)
    target=json.loads(Path(OUT,'prepare.json').read_text())
    prediction=float(np.exp(intercept+slope*np.log(target['scaling_parameters'])))
    result=dict(frozen_at_unix=time.time(),law='BPB=exp(intercept)*N_scaling^slope along D approximately 8*N_scaling',
                intercept=float(intercept),slope=float(slope),predicted_bpb=prediction,
                target=target,calibration=rows,fixed_eval_sha256=rows[0]['fixed_eval_sha256'],fit_rmse=float(np.sqrt(np.mean((np.exp(intercept+slope*x)-np.exp(y))**2))),
                limitation='Three sizes, one seed; joint model/data scaling, long extrapolation; not independent N,D exponents.',
                model_source_sha256=hashlib.sha256(Path('/root/a3/pico/fox_speedrun.py').read_bytes()).hexdigest())
    Path(OUT,'prediction.json').write_text(json.dumps(result,indent=2))
    volume.commit()
    return result


@app.function(image=image,volumes={'/vol':volume},gpu='H100!:8',cpu=(16,16),memory=(131072,131072),
              timeout=13500,retries=0,scaledown_window=2,max_containers=1,
              secrets=[modal.Secret.from_name('wandb-secret')])
def final():
    if not Path(OUT,'prediction.json').is_file():
        raise RuntimeError('Freeze compatible scaling prediction first')
    import hashlib
    prediction=json.loads(Path(OUT,'prediction.json').read_text())
    assert prediction['model_source_sha256']==hashlib.sha256(Path('/root/a3/pico/fox_speedrun.py').read_bytes()).hexdigest()
    assert prediction['fixed_eval_sha256']==hashlib.sha256(Path(BASE,'fixed_eval.pt').read_bytes()).hexdigest()
    smoke=json.loads(Path(OUT,'distributed-smoke-h100/result.json').read_text())
    assert 'H100' in smoke['gpu_name']
    # >10 excludes compile warmup, leaving 9 timed updates in the 20-update smoke.
    projected=smoke['train_seconds']/9*json.loads(Path(OUT,'prepare.json').read_text())['steps']
    if projected>10000:
        raise RuntimeError(f'Projected training alone {projected:.0f}s leaves insufficient evaluation/budget margin')
    return run_training('final-fox-d24',24,8,seconds=12900)


@app.function(image=image,volumes={'/vol':volume},cpu=.25,memory=1024,
              timeout=120,retries=0,scaledown_window=2,
              secrets=[modal.Secret.from_name('wandb-secret')])
def status():
    activate()
    import wandb
    records={str(p.relative_to(OUT)):json.loads(p.read_text()) for p in Path(OUT).rglob('result.json')}
    runs=wandb.Api().runs('jinhaozhang123-university-of-toronto/a3-picochat-pilot',filters={'group':'fox-speedrun-v1'})
    return dict(results=records,wandb=[dict(id=r.id,name=r.name,state=r.state,url=r.url,
                summary={k:v for k,v in dict(r.summary).items() if k in ('step','train/loss','val/bpb','calibration_bpb','test_bpb','tokens')}) for r in runs])


@app.local_entrypoint()
def main(stage: str='prepare'):
    dest=Path(__file__).parent/'speedrun_results'
    dest.mkdir(exist_ok=True)
    if stage=='prepare':
        result=prepare.remote()
    elif stage=='kernel-check':
        result=kernel_check.remote()
    elif stage=='prepare-eval':
        result=prepare_eval.remote()
    elif stage=='distributed-check':
        result=distributed_check.remote()
    elif stage=='h100-check':
        result=h100_check.remote()
    elif stage=='calibrate':
        result=calibrate.remote()
    elif stage=='recover-calibration':
        result=recover_calibration.remote()
    elif stage=='predict':
        result=predict.remote()
    elif stage=='final':
        result=final.remote()
    elif stage=='status':
        result=status.remote()
    else:
        raise ValueError(stage)
    (dest/f'{stage}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
