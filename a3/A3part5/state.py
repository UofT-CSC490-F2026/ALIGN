"""Small file helpers shared by the launcher and checkpoint evaluations."""
import json
import math
from pathlib import Path

VALIDATION_TOKENS = 80 * 524288
BATCH_SIZE = 8
CONTEXTS = (512, 2048)
RESULT_FOLDER = 'bpb-default-budget-v1'


def record(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def latest_checkpoint(folder, optimizers=False):
    """Select a checkpoint with readable metadata and all required files."""
    folder = Path(folder)
    for path in sorted(folder.glob('meta_*.json'), reverse=True):
        try:
            meta = json.loads(path.read_text())
            step = int(path.stem.split('_')[-1])
            files = [folder / f'model_{step:06d}.pt']
            if optimizers:
                files += [folder / f'optim_{step:06d}_rank{r}.pt' for r in range(8)]
            if meta['step'] == step and all(f.is_file() and f.stat().st_size for f in files):
                return step, meta
        except (ValueError, KeyError, OSError):
            continue
    raise FileNotFoundError(f'No complete checkpoint in {folder}')


def bpb_rows(path, steps):
    rows = json.loads(Path(path).read_text()) if Path(path).exists() else []
    seen = set()
    for row in rows:
        key = row['model'], row['context']
        if (key in seen or key[0] not in steps or key[1] not in CONTEXTS
                or row['checkpoint_step'] != steps[key[0]] or row['batch_size'] != BATCH_SIZE
                or row['nominal_positions'] != VALIDATION_TOKENS
                or row['batches'] != VALIDATION_TOKENS // (BATCH_SIZE * key[1])
                or not math.isfinite(row['bpb']) or row['bpb'] < 0):
            raise ValueError('Saved BPB results do not match the requested evaluation')
        seen.add(key)
    return rows


def saved_answers(path, spec):
    answers = json.loads(Path(path).read_text()) if Path(path).exists() else []
    if len(answers) > len(spec['questions']):
        raise ValueError('Saved question count exceeds the specification')
    for answer, question in zip(answers, spec['questions']):
        if any(answer.get(k) != v for k, v in question.items()) or not isinstance(answer.get('completion'), str):
            raise ValueError('Saved answers do not match the question specification')
    return answers
