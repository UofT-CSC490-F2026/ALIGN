"""Fixed evaluation data and a small-model scaling study for the saved nano model."""
import hashlib
import json
import math
from pathlib import Path
import time

from A3part5.state import record, latest_checkpoint

DEPTHS = (4, 6, 8)
SEEDS = (42, 43)
CONTEXT = 2048
EVAL_BATCH = 8
EVAL_STEPS = 64  # 1,048,576 target positions per fixed set.
ROOT = Path('/vol/runs/fox-d24-seed42-v1')
STUDY = ROOT / 'scaling-study-v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def train_command(depth, seed, resume_step=None):
    if depth not in DEPTHS or seed not in SEEDS:
        raise ValueError('Unplanned calibration configuration')
    command = ['torchrun', '--standalone', '--nproc_per_node=1', '-m', 'A3part5.scaling_train',
               f'--depth={depth}', '--max-seq-len=2048', '--target-param-data-ratio=8',
               '--device-batch-size=2', '--fp8', '--eval-every=-1',
               '--core-metric-every=-1', '--sample-every=-1', '--save-every=250',
               '--model-tag=calibration', f'--run=scaling-d{depth}-seed{seed}']
    if resume_step is not None:
        command.append(f'--resume-from-step={resume_step}')
    return command


def configurations():
    """Use native model counts and the pinned trainer's batch/horizon rule."""
    import torch
    from nanochat.gpt import GPTConfig
    from A3part5.fox import FoXGPT
    counts = {}
    for depth in (*DEPTHS, 12, 24):  # Depth 12 only defines the native batch reference.
        width = depth * 64
        with torch.device('meta'):
            model = FoXGPT(GPTConfig(n_layer=depth, n_embd=width, n_head=width // 128,
                                    n_kv_head=width // 128, sequence_len=CONTEXT))
        counts[depth] = model.num_scaling_params()
    nref = counts[12]['transformer_matrices'] + counts[12]['lm_head']
    result = {}
    for depth in (*DEPTHS, 24):
        count = counts[depth]
        n = count['transformer_matrices'] + count['lm_head']
        batch = 2 ** round(math.log2(2**19 * (n / nref)**.383))
        result[str(depth)] = dict(counts=count, scaling_parameters=n, global_batch=batch,
                                  steps=8*n // batch, tokens=(8*n // batch)*batch)
    return result


def prepare():
    """Save document-disjoint subsets, packed by the native best-fit loader."""
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    import nanochat.dataloader as loader_module
    from nanochat.dataset import parquets_iter_batched, list_parquet_files
    from nanochat.tokenizer import get_tokenizer

    STUDY.mkdir(exist_ok=True)
    source_files = ['scaling.py', 'scaling_train.py', 'fox.py', 'run_train.py']
    source_hashes = {f: sha256(Path(__file__).with_name(f)) for f in source_files}
    source_hashes['pico/variants.py'] = sha256(Path(__file__).parents[1] / 'pico/variants.py')
    config = configurations()
    original = read(ROOT / 'configuration.json')
    assert config['24']['counts'] == original['counts']
    assert config['24']['tokens'] == original['planned_tokens']
    final_dir = ROOT / 'base_checkpoints/fox-d24'
    step, metadata = latest_checkpoint(final_dir)
    assert step == config['24']['steps'] == 5571
    protocol = dict(depths=list(DEPTHS), seeds=list(SEEDS), context=CONTEXT,
                    configurations=config, fit_depths=list(DEPTHS),
                    final_fit_depths=list(DEPTHS), formula='BPB = exp(intercept) * N^slope',
                    parameter_definition='native transformer_matrices + lm_head',
                    data_ratio=8, loss_floor=False, evaluation_batch=EVAL_BATCH,
                    evaluation_steps=EVAL_STEPS, positions_per_set=EVAL_BATCH*CONTEXT*EVAL_STEPS,
                    document_pool_per_set=8192, split='SHA256(text) parity; deduplicated text hashes',
                    tokenizer_sha256=sha256('/vol/cache/tokenizer/tokenizer.pkl'),
                    nano_checkpoint_step=step, nano_metadata_sha256=sha256(final_dir / f'meta_{step:06d}.json'),
                    upstream_commit=original['upstream_commit'], source_hashes=source_hashes,
                    interpretation='Retrospective study: existing nano performance is known; nano excluded from fitting',
                    seed_scope='model initialization only; identical deterministic training document order',
                    training_validation='disabled in calibration; native BPB on fixed sets after training',
                    references=['https://arxiv.org/abs/2203.15556', 'https://arxiv.org/abs/2202.06387'])
    protocol_path = STUDY / 'protocol.json'
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError('Study protocol/source changed; refusing to mix results')
    record(protocol_path, protocol)
    if (STUDY / 'evaluation_manifest.json').exists():
        manifest = read(STUDY / 'evaluation_manifest.json')
        for label in ('A', 'B'):
            assert sha256(STUDY / f'eval_{label}.pt') == manifest[label]['sha256']
        return protocol

    tokenizer = get_tokenizer()
    pools = {'A': [], 'B': []}
    ids = {'A': [], 'B': []}
    seen = set()
    document_index = 0
    for texts in parquets_iter_batched('val'):
        for text in texts:
            key = hashlib.sha256(text.encode()).hexdigest()
            label = 'A' if int(key[-1], 16) % 2 == 0 else 'B'
            if text.strip() and key not in seen and len(pools[label]) < 8192:
                seen.add(key)
                pools[label].append(text)
                ids[label].append(dict(source_document_index=document_index, text_sha256=key))
            document_index += 1
        if all(len(v) == 8192 for v in pools.values()):
            break
    if not all(len(v) == 8192 for v in pools.values()):
        raise ValueError('Insufficient distinct validation documents')
    assert not ({r['text_sha256'] for r in ids['A']} & {r['text_sha256'] for r in ids['B']})
    manifest = dict(source_shard=list_parquet_files()[-1], documents=ids)
    native_files = loader_module.list_parquet_files
    try:
        for label in ('A', 'B'):
            parquet = STUDY / f'documents_{label}.parquet'
            pq.write_table(pa.table({'text': pools[label]}), parquet)
            loader_module.list_parquet_files = lambda *a, path=parquet, **kw: [str(path)]
            loader = loader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit(
                tokenizer, EVAL_BATCH, CONTEXT, 'val', device='cpu')
            batches = []
            for _ in range(EVAL_STEPS):
                x, y, state = next(loader)
                if state['epoch'] != 1:
                    raise ValueError('Evaluation pool exhausted; refusing repeated data')
                batches.append((x.clone(), y.clone()))
            path = STUDY / f'eval_{label}.pt'
            temp = path.with_suffix('.tmp')
            torch.save(batches, temp)
            temp.replace(path)
            manifest[label] = dict(sha256=sha256(path), positions=EVAL_BATCH*CONTEXT*EVAL_STEPS,
                                   document_pool_sha256=sha256(parquet))
    finally:
        loader_module.list_parquet_files = native_files
    record(STUDY / 'evaluation_manifest.json', manifest)
    record(STUDY / 'status.json', {'status': 'prepared'})
    print('RESULT fixed evaluation sets prepared', flush=True)
    return protocol


def evaluate_checkpoint(checkpoint_dir, step, destination, depth, seed):
    import torch
    from nanochat.common import compute_init
    from nanochat.tokenizer import get_token_bytes
    from A3part5.fox import install
    from nanochat.loss_eval import evaluate_bpb

    compute_init('cuda')
    install()
    from nanochat.checkpoint_manager import build_model
    destination = Path(destination)
    protocol = read(STUDY / 'protocol.json')
    assert sha256('/vol/cache/tokenizer/tokenizer.pkl') == protocol['tokenizer_sha256']
    manifest = read(STUDY / 'evaluation_manifest.json')
    model, _, meta = build_model(str(checkpoint_dir), step, torch.device('cuda'), 'eval')
    assert model.config.sequence_len == CONTEXT and model.config.n_layer == depth
    counts = model.num_scaling_params()
    expected = protocol['configurations'][str(depth)]
    assert counts == expected['counts']
    assert step == expected['steps'] and meta['total_batch_size'] == expected['global_batch']
    results = {}
    token_bytes = get_token_bytes(device='cuda')
    for label in ('A', 'B'):
        path = STUDY / f'eval_{label}.pt'
        assert sha256(path) == manifest[label]['sha256']
        batches = torch.load(path, map_location='cpu', weights_only=True)
        print(f'EVALUATING depth={depth} seed={seed} set={label} steps={len(batches)}', flush=True)
        result_path = destination / f'evaluation_{label}.json'
        if result_path.exists():
            previous = read(result_path)
            assert previous['eval_sha256'] == manifest[label]['sha256'] and previous['step'] == step
            results[label] = previous['bpb']
            continue
        def on_device():
            for i, (x, y) in enumerate(batches, 1):
                yield x.to('cuda'), y.to('cuda')
                if i % 16 == 0:
                    print(f'EVAL_PROGRESS depth={depth} seed={seed} set={label} batch={i}/{EVAL_STEPS}', flush=True)
        results[label] = evaluate_bpb(model, on_device(), len(batches), token_bytes)
        record(result_path, dict(bpb=results[label], step=step, eval_sha256=manifest[label]['sha256']))
        print(f'RESULT depth={depth} seed={seed} set={label} BPB={results[label]}', flush=True)
    result = dict(depth=depth, seed=seed, step=step, tokens=step*meta['total_batch_size'],
                  scaling_parameters=expected['scaling_parameters'], parameters=counts['total'], bpb=results,
                  eval_hashes={label: manifest[label]['sha256'] for label in ('A', 'B')})
    record(destination / 'result.json', result)
    print('RESULT checkpoint comparison saved', flush=True)
    return result


def power_fit(rows):
    import numpy as np
    x = np.log([r['scaling_parameters'] for r in rows])
    y = np.log([r['bpb']['A'] for r in rows])
    if len(set(x)) < 3 or not np.isfinite(y).all():
        raise ValueError('Fit needs three distinct sizes and finite positive BPB')
    slope, intercept = np.polyfit(x, y, 1)
    return dict(intercept=float(intercept), slope=float(slope))


def predict(law, n):
    return math.exp(law['intercept'] + law['slope'] * math.log(n))


def freeze_prediction():
    """Freeze before evaluating the held-out model; never fit to nano results."""
    rows = [read(STUDY / f'd{d}-seed{s}/result.json') for d in DEPTHS for s in SEEDS]
    manifest = read(STUDY / 'evaluation_manifest.json')
    expected_hashes = {label: manifest[label]['sha256'] for label in ('A', 'B')}
    protocol = read(STUDY / 'protocol.json')
    for row in rows:
        assert row['eval_hashes'] == expected_hashes
        expected = protocol['configurations'][str(row['depth'])]
        assert row['step'] == expected['steps'] and row['tokens'] == expected['tokens']
    target_depth = 24
    n = protocol['configurations'][str(target_depth)]['scaling_parameters']
    law = power_fit(rows)
    result = dict(law=law, predicted_A_bpb=predict(law, n), target_depth=target_depth,
                  fit_rows=rows, eval_hashes=expected_hashes, created_unix=time.time(),
                  interpretation='Retrospective scaling study; target results excluded from regression')
    result['seed_specific_predictions'] = {str(s): predict(power_fit([r for r in rows if r['seed'] == s]), n) for s in SEEDS}
    path = STUDY / 'nano_prediction.json'
    if path.exists():
        saved = read(path)
        for key in ('law', 'predicted_A_bpb', 'target_depth', 'fit_rows', 'eval_hashes'):
            if saved[key] != result[key]:
                raise ValueError('Frozen prediction inputs changed')
        return saved
    record(path, result)
    return result


def final_summary():
    rows = [read(STUDY / f'd{d}-seed{s}/result.json') for d in DEPTHS for s in SEEDS]
    nano = read(STUDY / 'nano/result.json')
    final = read(STUDY / 'nano_prediction.json')
    def errors(prediction, actual):
        return dict(predicted=prediction, actual=actual, error=actual-prediction,
                    relative_error_percent=100*(actual/prediction-1))
    result = dict(small_models=rows, nano=nano, prediction=final,
                  nano_comparison=errors(final['predicted_A_bpb'], nano['bpb']['A']),
                  set_B_transfer=[dict(depth=r['depth'], seed=r['seed'],
                                      **errors(predict(final['law'], r['scaling_parameters']), r['bpb']['B']))
                                  for r in [*rows, nano]],
                  limitations=['Joint size/data scaling, not separate causal effects',
                               'No fitted loss floor; extrapolation can be inaccurate',
                               'Three calibration sizes; no intermediate holdout; 17.4x extrapolation beyond largest scaling-parameter count',
                               'Two initialization seeds per small size; one nano seed',
                               'Seed-specific predictions are sensitivity checks, not calibrated confidence intervals',
                               'New fixed sets are not claimed disjoint from previously reported validation examples',
                               'Nano was already trained and its general performance was known before this study'])
    record(STUDY / 'summary.json', result)
    return result


if __name__ == '__main__':
    import sys
    run = Path(sys.argv[1])
    depth, seed = int(sys.argv[2]), int(sys.argv[3])
    folder = ROOT / 'base_checkpoints/fox-d24' if depth == 24 else run / 'base_checkpoints/calibration'
    step, _ = latest_checkpoint(folder)
    evaluate_checkpoint(folder, step, run, depth, seed)
