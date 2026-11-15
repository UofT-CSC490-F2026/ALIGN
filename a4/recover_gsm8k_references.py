"""Recover GSM8K training references locally; optional bounded public train download.

Never fetches test data, model files, credentials, or compute endpoints.
"""
import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path
from analyze_reward_signals import load_records
from arithmetic_reward import reference_calculations

TRAIN_URL = ('https://raw.githubusercontent.com/openai/grade-school-math/'
             '3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl')
FINAL = re.compile(r'#### (\-?[0-9\.\,]+)')


def normalize_question(text):
    """Exact case/punctuation-preserving match after NFKC and whitespace collapse."""
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def final_number(text):
    match = FINAL.search(text)
    return match.group(1).replace(',', '') if match else None


def fetch_train(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('refusing to replace an existing dataset')
    with urllib.request.urlopen(TRAIN_URL, timeout=30) as stream:
        data = stream.read(10_000_001)
    if len(data) > 10_000_000:
        raise ValueError('unexpectedly large dataset')
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != 7473 or not all(set(row) == {'question', 'answer'} for row in rows):
        raise ValueError('unexpected GSM8K train schema/size')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
    provenance = dict(url=TRAIN_URL, rows=len(rows), bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(), split='main/train',
                      note='Official OpenAI train JSONL underlying openai/gsm8k main; no test data fetched.')
    with path.with_suffix('.provenance.json').open('x', encoding='utf-8') as stream:
        json.dump(provenance, stream, indent=2)
    return provenance


def recover(records, train_rows, permutation=None):
    by_question = defaultdict(list)
    for index, row in enumerate(train_rows):
        by_question[normalize_question(row['question'])].append((index, row))
    result, issues = [], []
    counts = dict(input_records=len(records), matched=0, unmatched=0, ambiguous=0,
                  index_verified=0, index_mismatch=0, index_unknown=0,
                  historical_final_mismatch=0, annotation_coverage=0, valid_annotation_coverage=0)
    for row in records:
        key = dict(step=row['step'], problem=row['problem'])
        matches = by_question.get(normalize_question(row['question']), [])
        if len(matches) != 1:
            kind = 'unmatched' if not matches else 'ambiguous'
            counts[kind] += 1
            issues.append(dict(**key, kind=kind))
            continue
        index, original = matches[0]
        counts['matched'] += 1
        if permutation is None:
            state = 'index_unknown'
        else:
            problem = row['problem']
            state = 'index_verified' if 0 <= problem < len(permutation) and int(permutation[problem]) == index else 'index_mismatch'
        counts[state] += 1
        final_ok = final_number(row['reference']) == final_number(original['answer']) and final_number(original['answer']) is not None
        if not final_ok:
            counts['historical_final_mismatch'] += 1
        if state == 'index_mismatch' or not final_ok:
            issues.append(dict(**key, kind=state if state == 'index_mismatch' else 'historical_final_mismatch', dataset_index=index))
        counts['annotation_coverage'] += '<<' in original['answer']
        counts['valid_annotation_coverage'] += bool(reference_calculations(original['answer']))
        result.append(dict(row, historical_reference=row['reference'], reference=original['answer'],
                           recovery=dict(dataset_index=index, index_state=state, historical_final_matches=final_ok)))
    return result, dict(counts=counts, issues=issues,
                       duplicate_normalized_train_questions=sum(len(rows)>1 for rows in by_question.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollouts', nargs='+', type=Path)
    parser.add_argument('--train-file', type=Path, required=True)
    parser.add_argument('--download-train', action='store_true', help='explicitly download only the small public train JSONL if absent')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.download_train and not args.train_file.exists():
        fetch_train(args.train_file)
    train_data = args.train_file.read_bytes()
    train_rows = [json.loads(line) for line in train_data.splitlines()]
    if len(train_rows) != 7473 or not all(isinstance(r.get('question'), str) and isinstance(r.get('answer'), str) for r in train_rows):
        raise ValueError('CLI requires the 7,473-row main/train dataset; never use test labels')
    records, input_counts, input_issues = load_records(args.rollouts)
    try:
        import numpy as np
        permutation = np.random.default_rng(42).permutation(len(train_rows)).tolist()
        shuffle = dict(algorithm='numpy.random.default_rng(42).permutation', numpy_version=np.__version__, seed=42)
    except ImportError:
        permutation = None
        shuffle = dict(algorithm='not reproduced: numpy unavailable', seed=42)
    recovered, report = recover(records, train_rows, permutation)
    report.update(input_counts=input_counts, input_issues=input_issues, shuffle=shuffle,
                  train_rows=len(train_rows), train_sha256=hashlib.sha256(train_data).hexdigest(),
                  rollout_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.rollouts})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir/'recovered.jsonl').open('x', encoding='utf-8') as stream:
        for row in recovered:
            stream.write(json.dumps(row)+'\n')
    (args.output_dir/'recovery.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
