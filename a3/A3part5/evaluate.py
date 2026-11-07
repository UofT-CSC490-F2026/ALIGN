"""Saved-checkpoint BPB, question completion and optional native CORE evaluation."""
import json
from pathlib import Path
import torch
from nanochat.common import compute_init
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from A3part5.fox import install
from A3part5.state import (record, latest_checkpoint, bpb_rows, saved_answers,
                          VALIDATION_TOKENS, BATCH_SIZE, CONTEXTS, RESULT_FOLDER)


def checkpoint_steps(output, require_nano=True):
    steps = {'pico': 500}
    try:
        steps['nano'] = latest_checkpoint(output / 'base_checkpoints/fox-d24')[0]
    except FileNotFoundError:
        if require_nano:
            raise
    return steps


def load_saved(output, name, step):
    if name == 'pico':
        from pico.variants import build
        model = build('fox', seed=42)
        saved = torch.load(output / 'pico/model.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        return model, get_tokenizer()
    install()
    from nanochat.checkpoint_manager import build_model
    model, tokenizer, _ = build_model(str(output / 'base_checkpoints/fox-d24'), step,
                                      torch.device('cuda'), 'eval')
    return model, tokenizer


def answer_questions(model, tokenizer, output, question_file):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(question_file).read_text())
    answers = saved_answers(output / 'answers.json', spec)
    model.eval()
    model.config.sequence_len = 512
    model.window_sizes = model._compute_window_sizes(model.config)
    with torch.inference_mode():
        for question in spec['questions'][len(answers):]:
            prefix = tokenizer(question['prompt'], prepend='<|bos|>')
            ids = torch.tensor([prefix], device='cuda', dtype=torch.long)
            for _ in range(spec['max_new_tokens']):
                token = model(ids[:, -512:])[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat((ids, token), dim=1)
            answers.append(dict(question, completion=tokenizer.decode(ids[0, len(prefix):].tolist()), grade=None))
            record(output / 'answers.json', answers)
            print(f"RESULT question {question['id']} saved to {output}", flush=True)


def questions(output, extra=False):
    output = Path(output)
    spec_path = Path(__file__).with_name('evaluation_questions_extra.json' if extra else 'evaluation_questions.json')
    spec = json.loads(spec_path.read_text())
    steps = checkpoint_steps(output)
    destination = output / 'questions-extra-v1' if extra else output
    destination.mkdir(parents=True, exist_ok=True)
    if extra:
        saved_spec = destination / 'questions.json'
        if saved_spec.exists() and json.loads(saved_spec.read_text()) != spec:
            raise ValueError('Question specification changed for this run')
        record(saved_spec, spec)
    for name in ('pico', 'nano'):
        target = destination / name if extra or name == 'pico' else destination
        if len(saved_answers(target / 'answers.json', spec)) == len(spec['questions']):
            continue
        model, tokenizer = load_saved(output, name, steps[name])
        answer_questions(model, tokenizer, target, spec_path)
        del model
        torch.cuda.empty_cache()


def default_validation(output, models=('pico', 'nano')):
    output = Path(output)
    destination = output / RESULT_FOLDER
    destination.mkdir(parents=True, exist_ok=True)
    steps = checkpoint_steps(output, require_nano='nano' in models)
    rows = bpb_rows(destination / 'bpb.json', steps)
    completed = {(r['model'], r['context']) for r in rows}
    for name in models:
        if all((name, context) in completed for context in CONTEXTS):
            continue
        model, tokenizer = load_saved(output, name, steps[name])
        model.eval()
        for context in CONTEXTS:
            if (name, context) in completed:
                continue
            model.config.sequence_len = context
            model.window_sizes = model._compute_window_sizes(model.config)
            batches = VALIDATION_TOKENS // (BATCH_SIZE * context)
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, BATCH_SIZE, context, 'val')
            print(f'START {name} context={context} tokens={VALIDATION_TOKENS} batches={batches}', flush=True)
            def progress():
                for i, batch in enumerate(loader, 1):
                    yield batch
                    if i % 128 == 0:
                        print(f'PROGRESS {name} context={context} batches={i}/{batches}', flush=True)
            bpb = evaluate_bpb(model, progress(), batches, get_token_bytes(device='cuda'))
            rows.append(dict(model=name, checkpoint_step=steps[name], context=context,
                             batch_size=BATCH_SIZE, batches=batches,
                             nominal_positions=VALIDATION_TOKENS, bpb=bpb))
            record(destination / 'bpb.json', rows)
            print('RESULT ' + json.dumps(rows[-1]), flush=True)
        del model
        torch.cuda.empty_cache()
    if 'nano' in models:
        summarize(output, rows)
    return rows


def summarize(output, rows):
    """Save measured BPB and token accounting; scaling is a separate study."""
    output = Path(output)
    config = json.loads((output / 'configuration.json').read_text())
    result = dict(bpb=rows,
                  actual_training_tokens=checkpoint_steps(output)['nano'] * config['total_batch_size'])
    record(output / RESULT_FOLDER / 'evaluation.json', result)
    print('RESULT final BPB summary saved', flush=True)


def pico(output):
    """Train the reference only if no previous attempt or checkpoint exists."""
    from pico.train import train
    output = Path(output)
    if (output / 'result.json').exists():
        result = json.loads((output / 'result.json').read_text())
        if result['status'] != 'complete' or result['completed_steps'] != 500:
            raise RuntimeError('Incomplete pico training needs explicit recovery; refusing to retrain')
    else:
        if (output / 'training-started.json').exists() or (output / 'model.pt').exists():
            raise RuntimeError('Previous pico attempt exists; refusing to overwrite it')
        record(output / 'training-started.json', {'steps': 500})
        print('RESULT pico training attempt recorded', flush=True)
        train('fox', 42, str(output), steps=500, seconds=1200)
        result = json.loads((output / 'result.json').read_text())
        if result['completed_steps'] != 500:
            raise RuntimeError('Pico reference did not finish 500 steps')
    default_validation(output.parent, models=('pico',))


def nano(output):
    default_validation(output)
    questions(output)
    questions(output, extra=True)


def core(output):
    """Native full CORE on the final nano checkpoint; never inject a score."""
    from scripts.base_eval import evaluate_core
    output = Path(output)
    step = checkpoint_steps(output)['nano']
    target = output / 'core-final.json'
    if target.exists():
        previous = json.loads(target.read_text())
        if previous.get('checkpoint_step') != step or previous.get('max_per_task') != -1:
            raise ValueError('Existing CORE result belongs to a different evaluation')
        return
    model, tokenizer = load_saved(output, 'nano', step)
    model.eval()
    result = evaluate_core(model, tokenizer, torch.device('cuda'), max_per_task=-1)
    result.update(checkpoint_step=step, evaluation_context=model.config.sequence_len, max_per_task=-1)
    record(target, result)
    print('RESULT CORE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    import sys
    compute_init('cuda')
    {'pico': pico, 'nano': nano, 'extra-questions': lambda p: questions(p, extra=True),
     'default-validation': default_validation, 'core': core}[sys.argv[1]](sys.argv[2])
