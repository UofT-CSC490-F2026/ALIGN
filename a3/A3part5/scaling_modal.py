"""Detached calibration study; the existing depth-24 model is evaluated only."""
from pathlib import Path
import modal

app = modal.App('a3-part5-scaling')
volume = modal.Volume.from_name('a3-part5-fox')
secret = modal.Secret.from_name('a3-p4-wandb')
image = None
if modal.is_local():
    from A3part5.modal_app import image as original_image
    image = original_image
    for filename in ('modal_app.py', 'scaling.py', 'scaling_train.py'):
        image = image.add_local_file(str(Path(__file__).with_name(filename)), f'/root/a3/A3part5/{filename}')


def activate():
    import site
    site.addsitedir('/root/nanochat/.venv/lib/python3.12/site-packages')


@app.function(image=image, volumes={'/vol': volume}, cpu=4, memory=16384,
              timeout=1800, retries=0, max_containers=1, scaledown_window=2)
def prepare_study():
    activate()
    volume.reload()
    from A3part5.scaling import prepare
    try:
        return prepare()
    finally:
        volume.commit()


@app.function(image=image, volumes={'/vol': volume}, secrets=[secret],
              gpu='H100!', cpu=4, memory=32768, timeout=7200, retries=0,
              max_containers=1, scaledown_window=2)
def small_model(depth: int, seed: int):
    activate()
    volume.reload()
    from A3part5.modal_app import execute, train_environment
    from A3part5.scaling import STUDY, read, train_command
    from A3part5.state import record, latest_checkpoint
    output = STUDY / f'd{depth}-seed{seed}'
    output.mkdir(exist_ok=True)
    expected = read(STUDY / 'protocol.json')['configurations'][str(depth)]
    folder = output / 'base_checkpoints/calibration'
    try:
        if (output / 'result.json').exists():
            result = read(output / 'result.json')
            assert result['tokens'] == expected['tokens'] and result['seed'] == seed
            return result
        try:
            step, meta = latest_checkpoint(folder)
        except FileNotFoundError:
            step = None
            if (output / 'training-started.json').exists():
                raise RuntimeError('Started run has no complete checkpoint; refusing an automatic restart')
        if step is not None:
            assert meta['total_batch_size'] == expected['global_batch']
            assert step <= expected['steps']
            if step < expected['steps'] and not (folder / f'optim_{step:06d}_rank0.pt').is_file():
                raise RuntimeError('Missing optimizer state for training recovery')
        env = dict(train_environment(output), A3_SCALING_SEED=str(seed), PYTHONUNBUFFERED='1')
        if step != expected['steps']:
            record(output / 'training-started.json', dict(depth=depth, seed=seed, resume_from=step))
            record(STUDY / 'status.json', dict(status='training_calibration', depth=depth, seed=seed))
            volume.commit()
            execute(train_command(depth, seed, step), output, env)
        record(STUDY / 'status.json', dict(status='evaluating_calibration', depth=depth, seed=seed))
        volume.commit()
        execute(['python', '-m', 'A3part5.scaling', str(output), str(depth), str(seed)],
                output / 'evaluation-log', env)
        return read(output / 'result.json')
    except Exception as error:
        record(STUDY / 'status.json', dict(status='failed', depth=depth, seed=seed, error=str(error)))
        volume.commit()
        raise


@app.function(image=image, volumes={'/vol': volume}, secrets=[secret],
              gpu='H100!', cpu=4, memory=32768, timeout=3600, retries=0,
              max_containers=1, scaledown_window=2)
def nano_evaluation():
    activate()
    volume.reload()
    from A3part5.modal_app import execute, train_environment
    from A3part5.scaling import ROOT, STUDY, read
    from A3part5.state import record
    if not (STUDY / 'nano_prediction.json').is_file():
        raise RuntimeError('Freeze the small-model prediction before nano evaluation')
    output = STUDY / 'nano'
    output.mkdir(exist_ok=True)
    if (output / 'result.json').exists():
        return read(output / 'result.json')
    record(STUDY / 'status.json', {'status': 'evaluating_existing_nano'})
    volume.commit()
    env = dict(train_environment(ROOT), PYTHONUNBUFFERED='1')
    execute(['python', '-m', 'A3part5.scaling', str(output), '24', '42'], output / 'evaluation-log', env)
    return read(output / 'result.json')


@app.function(image=image, volumes={'/vol': volume}, cpu=1, memory=4096,
              timeout=86400, retries=0, max_containers=1, scaledown_window=2)
def pipeline(after_call: str = ""):
    # A handoff waits for the existing worker; never cancels or duplicates it.
    if after_call:
        try:
            modal.FunctionCall.from_id(after_call).get()
        except Exception:
            volume.reload()
            from A3part5.scaling import STUDY, DEPTHS, SEEDS, read
            # The old controller stops at its removed depth-12 configuration.
            # Continue only if every retained model finished and saved results.
            for depth in DEPTHS:
                for seed in SEEDS:
                    read(STUDY / f"d{depth}-seed{seed}/result.json")
    activate()
    from A3part5.scaling import STUDY, SEEDS, freeze_prediction, final_summary
    from A3part5.state import record
    try:
        prepare_study.remote()
        for depth in (4, 6, 8):
            for seed in SEEDS:
                small_model.remote(depth, seed)
        volume.reload()
        freeze_prediction()
        volume.commit()
        nano_evaluation.remote()
        volume.reload()
        result = final_summary()
        record(STUDY / 'status.json', {'status': 'complete'})
        volume.commit()
        return result
    except Exception as error:
        volume.reload()
        if STUDY.exists():
            record(STUDY / 'status.json', {'status': 'failed', 'error': str(error)})
            volume.commit()
        raise


@app.local_entrypoint()
def main(after_call: str = ""):
    call = pipeline.spawn(after_call)
    print(f'Submitted scaling study: {call.object_id}', flush=True)
    print('Outputs: a3-part5-fox /runs/fox-d24-seed42-v1/scaling-study-v1/', flush=True)
