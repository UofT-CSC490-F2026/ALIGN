"""Local/free analysis of saved rollout reward signals. Standard library only.

Malformed lines/records/samples are reported and excluded, never repaired silently.
Output contains aggregates only, not raw responses. No model or cloud imports.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pvariance
from xml.sax.saxutils import escape

from arithmetic_reward import arithmetic_reward, reference_calculations
from rewards import completion_reward

CONFIGS = ('baseline', 'completion', 'arithmetic', 'combined')


def correlation(xs, ys):
    if len(xs) < 2 or not pvariance(xs) or not pvariance(ys):
        return None
    mx, my = mean(xs), mean(ys)
    return mean((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(pvariance(xs) * pvariance(ys))


def load_records(paths):
    seen = {}
    records, issues = [], []
    counts = dict(lines=0, duplicate_records=0, conflicting_records=0, malformed_lines=0,
                  malformed_records=0, malformed_samples=0)
    for path in map(Path, paths):
        with path.open(encoding='utf-8-sig') as stream:
            for line_number, line in enumerate(stream, 1):
                counts['lines'] += 1
                location = f'{path.name}:{line_number}'
                try:
                    row = json.loads(line)
                except ValueError:
                    counts['malformed_lines'] += 1
                    issues.append(dict(location=location, kind='invalid JSON'))
                    continue
                if not (isinstance(row, dict) and type(row.get('step')) is int and type(row.get('problem')) is int
                        and isinstance(row.get('question'), str) and isinstance(row.get('reference'), str)
                        and isinstance(row.get('samples'), list) and row['samples']):
                    counts['malformed_records'] += 1
                    issues.append(dict(location=location, kind='invalid record schema'))
                    continue
                key = row['step'], row['problem']
                if key in seen:
                    kind = 'duplicate_records' if seen[key] == row else 'conflicting_records'
                    counts[kind] += 1
                    issues.append(dict(location=location, kind=kind, step=key[0], problem=key[1]))
                    continue
                samples = []
                for index, sample in enumerate(row['samples']):
                    # New shaped logs keep legacy reward binary; correctness has priority.
                    correct = sample.get('correctness', sample.get('reward')) if isinstance(sample, dict) else None
                    if not (isinstance(sample, dict) and isinstance(sample.get('text'), str)
                            and type(correct) in (int, float, bool) and correct in (0, 1)):
                        counts['malformed_samples'] += 1
                        issues.append(dict(location=location, kind='invalid sample', sample=index))
                        continue
                    samples.append(dict(text=sample['text'], correctness=float(correct)))
                if samples:
                    seen[key] = row
                    records.append(dict(source=path.name, step=key[0], problem=key[1],
                                        question=row['question'], reference=row['reference'], samples=samples))
    return records, counts, issues


def summarize(records, completion_weight=.15, arithmetic_weight=.20):
    # Validate weights even if the input has no valid responses.
    completion_reward('', completion_weight)
    arithmetic_reward('', '', arithmetic_weight)
    if not math.isfinite(completion_weight):
        raise ValueError('completion weight must be finite')
    groups = {name: [] for name in CONFIGS}
    correctness, completion, arithmetic = [], [], []
    annotated = 0
    for row in records:
        annotated += bool(reference_calculations(row['reference']))
        group = {name: [] for name in CONFIGS}
        for sample in row['samples']:
            correct = sample['correctness']
            a = completion_reward(sample['text'], completion_weight)
            b = arithmetic_reward(row['reference'], sample['text'], arithmetic_weight)
            correctness.append(correct)
            completion.append(a)
            arithmetic.append(b)
            for name, value in zip(CONFIGS, (correct, correct + a, correct + b, correct + a + b)):
                group[name].append(value)
        for name in CONFIGS:
            groups[name].append(group[name])
    n = len(correctness)
    incorrect = [i for i, correct in enumerate(correctness) if not correct]
    result = dict(records=len(records), responses=n,
                  records_with_nonstandard_sample_count=sum(len(row['samples']) != 16 for row in records),
                  samples_per_record={str(k): v for k, v in sorted(Counter(len(row['samples']) for row in records).items())},
                  records_with_reference_steps=annotated,
                  reference_step_coverage=annotated / len(records) if records else 0,
                  correctness_rate=mean(correctness) if n else None, configurations={})
    for name in CONFIGS:
        rewards = [r for group in groups[name] for r in group]
        bonuses = [r - c for r, c in zip(rewards, correctness)]
        variances = [pvariance(group) for group in groups[name]]
        result['configurations'][name] = dict(
            positive_reward_rate=mean(r > 0 for r in rewards) if n else None,
            mean_reward=mean(rewards) if n else None,
            mean_bonus=mean(bonuses) if n else None,
            incorrect_partial_credit_rate=mean(bonuses[i] > 0 for i in incorrect) if incorrect else None,
            incorrect_mean_bonus=mean(bonuses[i] for i in incorrect) if incorrect else None,
            mean_within_question_reward_variance=mean(variances) if variances else None,
            mean_within_question_advantage_variance=mean(variances) if variances else None,
            zero_advantage_question_rate=mean(v == 0 for v in variances) if variances else None,
            reward_correctness_correlation=correlation(rewards, correctness),
            bonus_correctness_correlation=correlation(bonuses, correctness),
        )
    result['mean_completion_bonus'] = mean(completion) if n else None
    result['mean_arithmetic_bonus'] = mean(arithmetic) if n else None
    return result


def analyze(paths, completion_weight=.15, arithmetic_weight=.20):
    records, counts, issues = load_records(paths)
    by_file = {Path(path).name: summarize([r for r in records if r['source'] == Path(path).name],
                                        completion_weight, arithmetic_weight) for path in paths}
    return dict(input_counts=counts, issues=issues, completion_weight=completion_weight,
                arithmetic_weight=arithmetic_weight,
                all=summarize(records, completion_weight, arithmetic_weight), by_file=by_file,
                limitations=['These are sampled training rollouts, not test-set accuracy.',
                             'Missing reference annotations cannot be reconstructed from a final answer.',
                             'Duplicate/conflicting (step, problem) records keep the first occurrence.',
                             'Variance is population variance, averaged equally over questions; centered advantages have the same variance.',
                             'Constant signals have undefined correlations, serialized as null.'])


def save_summary(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior analysis.
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    flat = []
    for source, result in [('all', report['all']), *report['by_file'].items()]:
        for config, metrics in result['configurations'].items():
            flat.append(dict(source=source, configuration=config, records=result['records'],
                             responses=result['responses'], correctness_rate=result['correctness_rate'], **metrics))
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    # A compact vector bar chart, no plotting dependency or online resources.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="260" viewBox="0 0 800 260">',
           '<rect width="800" height="260" fill="white"/>',
           '<text x="20" y="25" font-family="sans-serif" font-size="18">Positive training reward rate (not evaluation accuracy)</text>']
    for i, (name, metrics) in enumerate(report['all']['configurations'].items()):
        rate = metrics['positive_reward_rate'] or 0
        y = 50 + i * 45
        svg += [f'<text x="20" y="{y + 20}" font-family="sans-serif">{escape(name)}</text>',
                f'<rect x="130" y="{y}" width="{rate * 550:.3f}" height="28" fill="#3568a8"/>',
                f'<text x="{140 + rate * 550:.3f}" y="{y + 20}" font-family="sans-serif">{rate:.2%}</text>']
    svg.append('</svg>')
    (output / 'positive_rates.svg').write_text('\n'.join(svg), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path, required=True, help='new directory outside Git or ignored')
    parser.add_argument('--completion-weight', type=float, default=.15)
    parser.add_argument('--arithmetic-weight', type=float, default=.20)
    args = parser.parse_args()
    report = analyze(args.files, args.completion_weight, args.arithmetic_weight)
    save_summary(report, args.output_dir)
    print(json.dumps({k: report[k] for k in ('input_counts', 'all')}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
