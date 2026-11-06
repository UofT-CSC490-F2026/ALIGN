"""Minimal exact-source adapter: FoX class/cache plus logging/checkpoint instrumentation."""
import os
import json
import time
import hashlib
from pathlib import Path


def checkpoint(directory,step,model,optimizer,metadata,rank=0):
    import torch
    import torch.distributed as dist
    from nanochat.checkpoint_manager import save_checkpoint
    save_checkpoint(directory,step,model,optimizer,metadata,rank)
    Path(directory).mkdir(parents=True,exist_ok=True)
    torch.save({'metadata':metadata,'cpu_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state()},Path(directory)/f'resume_{step:06d}_rank{rank}.pt')
    if dist.is_initialized():
        dist.barrier()
    if rank==0:
        import modal
        modal.Volume.from_name('a3-picochat-pilot').commit()
    if dist.is_initialized():
        dist.barrier()


def finish(ns):
    import torch
    from nanochat.loss_eval import evaluate_bpb
    suite_path=Path(os.environ['NANOCHAT_BASE_DIR'],'fixed_eval.pt')
    batches=torch.load(suite_path,map_location='cpu',weights_only=True)
    fixed={}
    model=ns['orig_model']
    model.eval()
    with ns['disable_fp8'](model):
        for label,subset in [('calibration_bpb',batches[:16]),('test_bpb',batches[16:])]:
            local=subset[ns['ddp_rank']::ns['ddp_world_size']]
            device=model.get_device()
            fixed[label]=evaluate_bpb(model,((x.to(device),y.to(device)) for x,y in local),len(local),ns['token_bytes'])
    fixed['fixed_eval_sha256']=hashlib.sha256(suite_path.read_bytes()).hexdigest()
    if not ns['master_process']:
        return
    result=dict(status='complete',depth=ns['args'].depth,model=ns['model_config_kwargs'],
                parameters=ns['num_params'],scaling_parameters=ns['num_scaling_params'],
                steps=ns['step'],tokens=ns['total_tokens'],global_batch=ns['total_batch_size'],
                val_bpb=ns['val_bpb'],train_seconds=ns['total_training_time'],
                gpu_name=torch.cuda.get_device_name(),peak_memory_bytes=torch.cuda.max_memory_allocated(),
                elapsed_seconds=time.time()-float(os.environ['A3_RUN_START']),
                wandb_url=ns['wandb_run'].url,user_config=ns['user_config'],
                model_source_sha256=hashlib.sha256(Path(__file__).with_name('fox_speedrun.py').read_bytes()).hexdigest())
    result.update(fixed)
    if ns['args'].run=='final-fox-d24':
        prediction_path=Path(os.environ['A3_RUN_DIR']).parent/'prediction.json'
        prediction=json.loads(prediction_path.read_text())
        assert prediction['fixed_eval_sha256']==fixed['fixed_eval_sha256']
        result.update(predicted_bpb=prediction['predicted_bpb'],
                      prediction_error=fixed['calibration_bpb']-prediction['predicted_bpb'],
                      prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest())
    Path(os.environ['A3_RUN_DIR'],'result.json').write_text(json.dumps(result,indent=2))
    ns['wandb_run'].summary.update({k:v for k,v in result.items() if k not in ('model','user_config')})


def source_adapter(source):
    patches=[
        ('from nanochat.gpt import GPT, GPTConfig, Linear',
         'from nanochat.gpt import GPTConfig, Linear\nfrom pico.fox_speedrun import FoxGPT as GPT, install_cache\ninstall_cache()'),
        ('from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint',
         'from nanochat.checkpoint_manager import load_checkpoint\nfrom pico.speedrun_train import checkpoint as save_checkpoint'),
        ('wandb.init(project="nanochat", name=args.run, config=user_config)',
         'wandb.init(project="a3-picochat-pilot", group="fox-speedrun-v1", name=args.run, config={**user_config,"variant":"fox","training_shards":170})'),
        ('wandb_run.finish() # wandb run finish',
         'from pico.speedrun_train import finish\nfinish(globals())\nwandb_run.finish() # wandb run finish'),
    ]
    for old,new in patches:
        if source.count(old)!=1:
            raise RuntimeError(f'Source drift: {old}')
        source=source.replace(old,new)
    return source


if __name__=='__main__':
    import torch
    torch._dynamo.config.recompile_limit=64
    source=Path('/root/nanochat/scripts/base_train.py').read_text()
    revised=source_adapter(source)
    directory=Path(os.environ['A3_RUN_DIR'])
    directory.mkdir(parents=True,exist_ok=True)
    if int(os.environ.get('RANK','0'))==0:
        (directory/'upstream_base_train.py').write_text(source)
        (directory/'adapted_base_train.py').write_text(revised)
        (directory/'fox_speedrun.py').write_text(Path(__file__).with_name('fox_speedrun.py').read_text())
    os.environ['A3_RUN_START']=str(time.time())
    exec(compile(revised,'/root/nanochat/scripts/base_train.py','exec'),{'__name__':'__main__'})
