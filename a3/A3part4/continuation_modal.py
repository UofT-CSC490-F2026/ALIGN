"""Detached inference-only comparison of identical passage endings."""
from pathlib import Path
import re
import modal

app = modal.App('a3-part4-continuation')
volume = modal.Volume.from_name('a3-part4-baseline')
image = None
if modal.is_local():
    from A3part4.modal_app import image as original_image
    image = original_image.add_local_file(str(Path(__file__).with_name('continuation_eval.py')),
                                          '/root/a3/A3part4/continuation_eval.py')


@app.function(image=image, volumes={'/vol': volume}, gpu='H100', cpu=4, memory=16384,
              timeout=1800, retries=0, max_containers=1, scaledown_window=2)
def evaluate_saved(run_id: str):
    import site
    site.addsitedir('/root/nanochat/.venv/lib/python3.12/site-packages')
    volume.reload()
    from A3part4.continuation_eval import evaluate
    run_dir = Path('/vol/runs') / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_id)
    return evaluate(run_dir, volume.commit)


@app.local_entrypoint()
def main(run_id: str = 'baseline-context-seed42-v1'):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    call = evaluate_saved.spawn(run_id)
    print(f'Submitted matched continuation evaluation: {call.object_id}', flush=True)
    print(f'Output: a3-part4-baseline /runs/{run_id}/matched-continuation-v1/', flush=True)
