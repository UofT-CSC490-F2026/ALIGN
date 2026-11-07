"""Score identical passage endings with short and long preceding context."""
import hashlib
import json
from pathlib import Path

CONTEXTS = (512, 2048)
STEPS = (500, 1000)
DOCUMENTS = 256
TARGET_TOKENS = 128
FOLDER = 'matched-continuation-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def window(tokens, bos, context):
    """One BOS plus text; score only the shared final 128 targets."""
    if context not in CONTEXTS or len(tokens) != max(CONTEXTS):
        raise ValueError('Unexpected document or context length')
    sequence = [bos] + tokens[-context:]
    return sequence[:-1], [-1] * (context - TARGET_TOKENS) + sequence[-TARGET_TOKENS:]


def examples(tokenizer):
    from nanochat.dataset import parquets_iter_batched
    selected, seen = [], set()
    index = 0
    for texts in parquets_iter_batched('val'):
        encoded = tokenizer.encode(texts)
        for text, tokens in zip(texts, encoded):
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if len(tokens) >= max(CONTEXTS) and text_hash not in seen:
                selected.append(dict(index=len(selected), source_document_index=index,
                                     text_sha256=text_hash, tokens=tokens[:max(CONTEXTS)]))
                seen.add(text_hash)
                if len(selected) == DOCUMENTS:
                    return selected
            index += 1
    raise ValueError('Insufficient distinct held-out long documents')


def completed_rows(path, docs):
    rows = json.loads(path.read_text()) if path.exists() else []
    seen = set()
    for row in rows:
        key = row['checkpoint_step'], row['context'], row['index']
        if (key in seen or key[0] not in STEPS or key[1] not in CONTEXTS
                or not 0 <= key[2] < len(docs)
                or row['text_sha256'] != docs[key[2]]['text_sha256']):
            raise ValueError('Saved continuation scores do not match the protocol')
        seen.add(key)
    return rows


def summarize(rows):
    import numpy as np
    assert len(rows) == len(STEPS) * len(CONTEXTS) * DOCUMENTS
    by_key = {(r['checkpoint_step'], r['context'], r['index']): r for r in rows}
    weights = np.array([by_key[500, 512, i]['target_bytes'] for i in range(DOCUMENTS)])
    scores = {}
    for step in STEPS:
        for context in CONTEXTS:
            group = [by_key[step, context, i] for i in range(DOCUMENTS)]
            assert all(r['target_bytes'] == weights[i] for i, r in enumerate(group))
            scores[step, context] = np.array([r['bpb'] for r in group])
    rng = np.random.default_rng(42)
    bootstrap = rng.integers(0, DOCUMENTS, size=(2000, DOCUMENTS))
    def paired(delta):
        samples = ((delta * weights)[bootstrap].sum(axis=1) / weights[bootstrap].sum(axis=1))
        return dict(byte_weighted_difference=float(np.average(delta, weights=weights)),
                    paired_document_bootstrap_95_interval=np.quantile(samples, [.025, .975]).tolist())
    return dict(
        bpb=[dict(checkpoint_step=s, context=c, target_only_bpb=float(np.average(scores[s, c], weights=weights)))
             for s in STEPS for c in CONTEXTS],
        checkpoint2_minus_checkpoint1={str(c): paired(scores[1000, c] - scores[500, c]) for c in CONTEXTS},
        context_gain_512_minus_2048={str(s): paired(scores[s, 512] - scores[s, 2048]) for s in STEPS},
        change_in_context_gain=paired((scores[1000, 512]-scores[1000, 2048])-(scores[500, 512]-scores[500, 2048])),
        documents=DOCUMENTS, target_positions=DOCUMENTS*TARGET_TOKENS,
        scored_bytes=int(weights.sum()),
        limitations=['First eligible held-out long documents; not a random sample of all text',
                     'Validation corpus was already used for training monitoring',
                     'Context-window changes also change native SSSL window sizes',
                     'Checkpoint 2 received more training and a restarted learning-rate schedule',
                     'Bootstrap covers documents, not training-seed variability',
                     'Adaptation of sliding-window language-model evaluation, not an official benchmark score'])


def evaluate(run_dir, commit):
    import torch
    from nanochat.common import compute_init
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.checkpoint_manager import load_checkpoint
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.dataset import list_parquet_files
    from pico.variants import build
    from A3part4.train import write_json

    run_dir = Path(run_dir)
    output = run_dir / FOLDER
    output.mkdir(exist_ok=True)
    protocol = dict(documents=DOCUMENTS, contexts=list(CONTEXTS), target_tokens=TARGET_TOKENS,
                    selection='first distinct validation documents with at least 2048 tokens; first 2048 tokens',
                    scoring='native evaluate_bpb with all but final 128 targets masked to -1',
                    tokenizer_sha256=digest('/vol/cache/tokenizer/tokenizer.pkl'),
                    checkpoint_sha256={str(s): digest(run_dir/'checkpoints'/f'model_{s:06d}.pt') for s in STEPS},
                    source_sha256=digest(__file__), validation_shard=list_parquet_files()[-1],
                    reference='https://aclanthology.org/2021.acl-long.427/')
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing protocol differs; refusing to mix evaluations')
    write_json(protocol_path, protocol)
    commit()
    try:
        tokenizer = get_tokenizer()
        path = output / 'examples.json'
        if not path.exists():
            write_json(output / 'status.json', {'status': 'preparing_examples'})
            commit()
            write_json(path, examples(tokenizer))
            write_json(output / 'examples-sha256.json', {'sha256': digest(path)})
            commit()
        docs = json.loads(path.read_text())
        assert len(docs) == DOCUMENTS
        assert digest(path) == json.loads((output/'examples-sha256.json').read_text())['sha256']
        rows_path = output / 'scores.json'
        rows = completed_rows(rows_path, docs)
        done = {(r['checkpoint_step'], r['context'], r['index']) for r in rows}
        if len(done) < DOCUMENTS * 4:
            compute_init('cuda')
            token_bytes = get_token_bytes(device='cuda')
            bos = tokenizer.get_bos_token_id()
            for step in STEPS:
                if all((step, c, d['index']) in done for c in CONTEXTS for d in docs):
                    continue
                weights, _, _ = load_checkpoint(str(run_dir/'checkpoints'), step, device='cuda', load_optimizer=False)
                for context in CONTEXTS:
                    pending = [d for d in docs if (step, context, d['index']) not in done]
                    if not pending:
                        continue
                    model = build('baseline', seed=42, seq_len=context, vocab_size=tokenizer.get_vocab_size())
                    model.load_state_dict(weights, strict=True)
                    model.eval()
                    print(f'EVALUATING checkpoint={step} context={context} documents={len(pending)}', flush=True)
                    for document in pending:
                        x, y = window(document['tokens'], bos, context)
                        x = torch.tensor([x], device='cuda')
                        y = torch.tensor([y], device='cuda')
                        target_bytes = int(token_bytes[y[0, -TARGET_TOKENS:]].sum().item())
                        if target_bytes <= 0:
                            raise ValueError('Continuation contains no scored bytes')
                        bpb = evaluate_bpb(model, [(x, y)], 1, token_bytes)
                        rows.append(dict(checkpoint_step=step, context=context, index=document['index'],
                                         text_sha256=document['text_sha256'], target_bytes=target_bytes, bpb=bpb))
                        write_json(rows_path, rows)
                        if len(rows) % 16 == 0:
                            write_json(output/'status.json', dict(status='evaluating', completed=len(rows), total=DOCUMENTS*4))
                            commit()
                            print(f'PROGRESS {len(rows)}/{DOCUMENTS*4}', flush=True)
                    commit()
                    del model
                del weights
        result = summarize(rows)
        write_json(output/'summary.json', result)
        write_json(output/'status.json', {'status': 'complete'})
        commit()
        return result
    except Exception as error:
        write_json(output/'status.json', dict(status='failed', error=str(error)))
        commit()
        raise
