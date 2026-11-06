"""Fixed held-out evaluations; no training data or adaptive question selection."""
import hashlib
import json
import math
from pathlib import Path
import torch
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.dataset import parquets_iter_batched


def suite(tokenizer, path):
    path = Path(path)
    if path.exists():
        data = torch.load(path, map_location='cpu', weights_only=False)
        return data
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, 512, 'val', device='cpu')
    packed = []
    for _ in range(256):
        x, y = next(loader)
        packed.append((x.clone(), y.clone()))
    documents = []
    for texts in parquets_iter_batched('val'):
        for text in texts:
            tokens = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
            if len(tokens) >= 2049:
                documents.append(tokens[:2049])
                if len(documents) == 48:
                    break
        if len(documents) == 48:
            break
    if len(documents) != 48:
        raise RuntimeError('Insufficient long held-out documents')
    data = dict(packed=packed, documents=documents,
                split='shard_06542.parquet', version=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


@torch.no_grad()
def bpb(model, data, token_bytes, split='calibration'):
    model.eval()
    pairs = data['packed'][:128] if split == 'calibration' else data['packed'][128:]
    nats, nbytes = 0., 0
    for x, y in pairs:
        x, y = x.to('cuda'), y.to('cuda')
        losses = model(x, y, loss_reduction='none').reshape_as(y)
        sizes = token_bytes[y]
        nats += (losses * (sizes > 0)).sum().item()
        nbytes += sizes.sum().item()
    return nats / (math.log(2) * nbytes)


@torch.no_grad()
def suffix_eval(model, data, token_bytes, context):
    """Same final 256 target tokens from each document at both context sizes."""
    model.eval()
    rows = []
    for i, tokens in enumerate(data['documents']):
        t = torch.tensor(tokens, device='cuda')[None]
        x, y = t[:, 2048-context:2048], t[:, 2049-context:2049]
        losses = model(x, y, loss_reduction='none').reshape_as(y)[:, -256:]
        sizes = token_bytes[y[:, -256:]]
        nats = (losses * (sizes > 0)).sum().item()
        nbytes = sizes.sum().item()
        rows.append(dict(document=i, nats=nats, bytes=nbytes,
                         bpb=nats / (math.log(2) * nbytes)))
    return {'context': context, 'target_tokens': len(rows)*256,
            'bpb': sum(r['nats'] for r in rows)/(math.log(2)*sum(r['bytes'] for r in rows)),
            'documents': rows}


