"""Part 4 correction: original Part 2 baseline checkpoints, native upstream GPT."""
import json
from pathlib import Path
import modal

if modal.is_local():
    from pico_modal import image
else:
    image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('a3-part4-baseline-context')
volume=modal.Volume.from_name('a3-picochat-pilot')
ROOT='/vol/results/part4-baseline-v1'


@app.function(image=image,volumes={'/vol':volume},gpu='H100',cpu=4,memory=16384,
              timeout=3600,retries=0,scaledown_window=2,max_containers=1,
              secrets=[modal.Secret.from_name('wandb-secret')])
def experiment():
    import site
    site.addsitedir('/root/nanochat/.venv/lib/python3.12/site-packages')
    import hashlib
    import torch
    import wandb
    from nanochat.common import compute_init
    from nanochat.tokenizer import get_tokenizer,get_token_bytes
    from pico.context_train import BASE,load_model,train,save_json
    from pico.context_eval import suite,suffix_eval
    import nanochat.dataset as dataset
    compute_init('cuda')
    assert dataset.DATA_DIR=='/vol/cache/base_data_climbmix'
    assert sorted(p.name for p in Path(dataset.DATA_DIR).glob('*.parquet')) == [f'shard_{i:05d}.parquet' for i in (0,1,2,3,6542)]
    assert hashlib.sha256(Path('/vol/cache/tokenizer/tokenizer.pkl').read_bytes()).hexdigest()=='a228803c68627f650e640c819e71ea0b90f49a96118858ed781a00a5ba63b6bf'
    dest=Path(ROOT)
    if dest.exists():
        raise RuntimeError('Attempt exists; inspect rather than overwrite/repeat')
    dest.mkdir(parents=True)
    save_json(dest/'started.json',{'implementation':'native nanochat.gpt.GPT','sources':'original pilot-v1 baseline','timeout_seconds':3600})
    volume.commit()
    data=suite(get_tokenizer(),BASE/'eval_suite.pt')
    byte_lengths=get_token_bytes(device='cuda')
    originals=[]
    results=[]
    cross=[]
    source_hashes={}
    for seed in (42,43):
        source=Path(f'/vol/results/pilot-v1/baseline/seed{seed}/model.pt')
        source_hashes[str(seed)]=hashlib.sha256(source.read_bytes()).hexdigest()
        for seq in (512,2048):
            model,parent=load_model(source,seq,native_baseline=True)
            if seq==512:
                adapted,_=load_model(source,seq)
                x=torch.randint(0,32768,(1,64),device='cuda')
                with torch.no_grad():
                    torch.testing.assert_close(model(x),adapted(x),atol=2e-5,rtol=2e-4)
                del adapted
            originals.append(dict(seed=seed,context=seq,suffix_eval=suffix_eval(model,data,byte_lengths,seq)))
            del model
            torch.cuda.empty_cache()
        save_json(dest/'originals.json',originals)
        volume.commit()
        for seq in (512,2048):
            name=f'context-baseline-{seq}-seed{seed}'
            row=train(name,seq=seq,steps=1024,seed=seed,source=source,seconds=650,
                      commit=volume.commit,native_baseline=True,output_base=dest,
                      run_group='part4-baseline-v1')
            if row['status']!='complete':
                raise RuntimeError('Partial training; no automatic retry')
            results.append(row)
            for tested in (512,2048):
                model,_=load_model(dest/name/'model.pt',tested,native_baseline=True)
                cross.append(dict(seed=seed,training_context=seq,evaluation_context=tested,
                                  suffix_eval=suffix_eval(model,data,byte_lengths,tested)))
                del model
                torch.cuda.empty_cache()
            save_json(dest/'cross-eval.json',cross)
            volume.commit()
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hashes[str(seed)]
    audit=[]
    api=wandb.Api()
    for row in results:
        parts=row['wandb_url'].split('/')
        run=api.run('/'.join((parts[-4],parts[-3],parts[-1])))
        losses=list(run.scan_history(keys=['train/loss']))
        vals=list(run.scan_history(keys=['val/calibration_bpb']))
        if not losses or not vals:
            raise RuntimeError('Missing W&B curves')
        audit.append(dict(name=row['name'],state=run.state,loss_points=len(losses),validation_points=len(vals)))
    output=dict(originals=originals,results=results,cross_eval=cross,source_sha256=source_hashes,
                wandb_audit=audit,eval_suite_sha256=hashlib.sha256((BASE/'eval_suite.pt').read_bytes()).hexdigest())
    save_json(dest/'comparison.json',output)
    volume.commit()
    return output


@app.local_entrypoint()
def main():
    result=experiment.remote()
    dest=Path(__file__).parent/'baseline_context_results'
    dest.mkdir(exist_ok=True)
    (dest/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Saved {dest / "comparison.json"}')
