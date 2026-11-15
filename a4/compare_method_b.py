"""Local-only, training-only comparison of Method B candidates and learning signals.

Input is the recovered JSONL produced by recover_gsm8k_references.py. Outputs are
local aggregate JSON/CSV plus raw audit examples and scored data under an ignored
folder. Does not fetch data, load models, or import Modal/torch.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pvariance
from analyze_reward_signals import correlation
from recover_gsm8k_references import final_number
from method_b_candidates import candidate_signals
from part3_failure_labels import failure_kind, NUMBER
from rewards import completion_reward

CONFIGS = ('baseline', 'A', 'B1', 'A+B1', 'B2', 'A+B2', 'B3', 'A+B3', 'B4', 'A+B4', 'B5', 'A+B5')


def varying(values):
    return max(values) - min(values) > 1e-12


def safe_failure(question, reference, text, correct):
    if correct:
        return 'correct (original rollout label)'
    try:
        return failure_kind(question, reference, text)
    except (ValueError, OverflowError):
        return 'heuristic unavailable: invalid numeric magnitude'


def score_records(records):
    groups, flat = [], []
    for record in records:
        if record.get('recovery', {}).get('index_state') != 'index_verified' or not record['recovery']['historical_final_matches']:
            continue
        group = []
        question, reference = record['question'], record['reference']
        words, quantities = len(question.split()), len(NUMBER.findall(question))
        for index, sample in enumerate(record['samples']):
            text, correct = sample['text'], sample['correctness']
            signals = candidate_signals(question, reference, text)
            a = completion_reward(text)
            bonuses = dict(baseline=0., A=a)
            for name in ('B1', 'B2', 'B3', 'B4', 'B5'):
                bonuses[name] = signals[name]
                bonuses['A+'+name] = a + signals[name]
            item = dict(step=record['step'], problem=record['problem'], sample=index, correct=correct,
                        group_size=len(record['samples']), source=record['source'],
                        text=text, question=question, reference=reference, bonus=bonuses,
                        recomputed_correct=int(final_number(text) == final_number(reference)),
                        diagnostics=signals['diagnostics'],
                        failure=safe_failure(question, reference, text, correct),
                        step_bin=f'{min(record["step"] // 117, 3)*117}-{min((record["step"]//117+1)*117-1,466)}',
                        length_bin='<30' if words < 30 else '30-49' if words < 50 else '50-79' if words < 80 else '80+',
                        quantity_bin='0-1' if quantities <= 1 else '2-3' if quantities <= 3 else '4-5' if quantities <= 5 else '6+')
            item['id'] = f'{item["step"]}:{item["problem"]}:{index}'
            flat.append(item)
            group.append(item)
        groups.append(group)
    return groups, flat


def sample_metrics(items, config):
    n = len(items)
    if not n:
        return dict(responses=0)
    bonuses = [r['bonus'][config] for r in items]
    correctness = [r['correct'] for r in items]
    totals = [c+b for c,b in zip(correctness,bonuses)]
    wrong = [i for i,c in enumerate(correctness) if not c]
    right = [i for i,c in enumerate(correctness) if c]
    return dict(responses=n, correctness_rate=mean(correctness),
                nonzero_shaping_rate=mean(abs(b)>1e-12 for b in bonuses),
                positive_bonus_rate=mean(b>1e-12 for b in bonuses),
                negative_bonus_rate=mean(b < -1e-12 for b in bonuses),
                positive_total_rate=mean(r>1e-12 for r in totals),
                mean_bonus=mean(bonuses), mean_total=mean(totals),
                incorrect_partial_credit_rate=mean(bonuses[i]>1e-12 for i in wrong) if wrong else None,
                incorrect_penalty_rate=mean(bonuses[i]<-1e-12 for i in wrong) if wrong else None,
                correct_positive_bonus_rate=mean(bonuses[i]>1e-12 for i in right) if right else None,
                correct_penalty_rate=mean(bonuses[i]<-1e-12 for i in right) if right else None,
                bonus_correctness_correlation=correlation(bonuses,correctness),
                total_correctness_correlation=correlation(totals,correctness))


def group_metrics(groups, config):
    if not groups:
        return dict(groups=0)
    all_wrong = [g for g in groups if not any(r['correct'] for r in g)]
    totals = lambda g,c: [r['correct']+r['bonus'][c] for r in g]
    gained = sum(varying(totals(g,config)) for g in all_wrong)
    a_zero = [g for g in all_wrong if not varying(totals(g,'A'))]
    return dict(groups=len(groups), all_wrong_groups=len(all_wrong),
                all_wrong_gaining_variance=gained,
                all_wrong_gaining_variance_rate=gained/len(all_wrong) if all_wrong else None,
                zero_advantage_group_rate=mean(not varying(totals(g,config)) for g in groups),
                mean_within_group_variance=mean(pvariance(totals(g,config)) for g in groups),
                all_wrong_A_zero_groups=len(a_zero),
                additional_all_wrong_differentiated_beyond_A=sum(varying(totals(g,config)) for g in a_zero))


def summarize(groups):
    flat = [item for g in groups for item in g]
    return {config: dict(**sample_metrics(flat,config), **group_metrics(groups,config)) for config in CONFIGS}


def select_audit(flat):
    """Fixed stratified sample for manual review, not a precision benchmark."""
    strata = {
        'B1_credit_wrong': lambda r: r['bonus']['B1']>0 and not r['correct'],
        'B2_only_credit': lambda r: r['bonus']['B2']>0 and r['bonus']['B1']==0,
        'B2_no_credit_with_tools': lambda r: r['bonus']['B2']==0 and '<|python_start|>' in r['text'],
        'B2_no_credit_no_tools': lambda r: r['bonus']['B2']==0 and '<|python_start|>' not in r['text'],
        'B3_rejects_B2': lambda r: r['bonus']['B2']>r['bonus']['B3'],
        'B4_credit_wrong': lambda r: r['bonus']['B4']>0 and not r['correct'],
        'B5_penalty': lambda r: r['bonus']['B5']<0,
    }
    rng = random.Random(490)
    result = []
    for name, predicate in strata.items():
        eligible = [r for r in flat if predicate(r)]
        for row in rng.sample(eligible,min(10,len(eligible))):
            result.append(dict(stratum=name, **row))
    return result


def compare(records):
    groups, flat = score_records(records)
    clean16 = [g for g in groups if len(g)==16]
    label_clean = [g for g in clean16 if all(r['correct']==r['recomputed_correct'] for r in g)]
    diagnostics = dict(responses=len(flat), groups=len(groups),
        correctness_label_disagreements=sum(r['correct']!=r['recomputed_correct'] for r in flat),
        B2_greater_than_B1=sum(r['bonus']['B2']>r['bonus']['B1']+1e-12 for r in flat),
        B2_less_than_B1=sum(r['bonus']['B2']<r['bonus']['B1']-1e-12 for r in flat),
        B2_only_credit=sum(r['bonus']['B2']>0 and r['bonus']['B1']==0 for r in flat),
        B3_rejects_B2=sum(r['bonus']['B2']>r['bonus']['B3']+1e-12 for r in flat),
        ungated_consistency_responses=sum(r['diagnostics']['ungated_consistency'] for r in flat),
        ungated_consistency_wrong=sum(r['diagnostics']['ungated_consistency'] and not r['correct'] for r in flat),
        B4_wrong=sum(r['bonus']['B4']>0 and not r['correct'] for r in flat),
        incomplete_tool=sum(r['diagnostics']['incomplete_tool'] for r in flat),
        repeated_digit=sum(r['diagnostics']['repeated_digit'] for r in flat),
        termination_metadata='Not present in these historical exports; termination versus token-limit truncation is unknown.',
        failure_counts=dict(Counter(r['failure'] for r in flat)),
        reference_supported_groups=sum(g[0]['diagnostics']['reference_steps']>0 for g in groups),
        unused_numeric_value_flag=sum(r['diagnostics']['unused_numeric_values']>0 for r in flat),
        unused_numeric_value_flag_correct=sum(r['correct'] and r['diagnostics']['unused_numeric_values']>0 for r in flat),
        ungrounded_operand_flag=sum(r['diagnostics']['ungrounded_operand_values']>0 for r in flat),
        ungrounded_operand_flag_correct=sum(r['correct'] and r['diagnostics']['ungrounded_operand_values']>0 for r in flat),
        B5_penalized_correct=sum(r['correct'] and r['bonus']['B5']<0 for r in flat))
    breakdowns = []
    for population, selected_groups in [('all_retained', groups), ('primary_exactly16_labels_agree', label_clean)]:
        selected_flat = [r for g in selected_groups for r in g]
        for dimension in ('step_bin','length_bin','quantity_bin','failure','step'):
            bins = defaultdict(list)
            for item in selected_flat:
                bins[str(item[dimension])].append(item)
            for name, rows in sorted(bins.items()):
                keys = {(r['step'],r['problem']) for r in rows}
                subset_groups = [g for g in selected_groups if (g[0]['step'],g[0]['problem']) in keys] if dimension!='failure' else []
                for config in CONFIGS:
                    breakdowns.append(dict(population=population,dimension=dimension,bin=name,configuration=config,
                                           **sample_metrics(rows,config),
                                           **(group_metrics(subset_groups,config) if subset_groups else {})))
    report = dict(definitions=dict(B1='Unchanged production strict expression-tree match, +0.20 max',
        B2='B1 with child sorting at Add/Mult only, +0.20 max',
        B3='B2 with numeric provenance closure from question/structural constants, +0.20 max; diagnostic',
        B4='Verified reference-aligned last tool result numerically equals valid final answer, +0.05; diagnostic',
        B5='-0.05 for >=12 identical consecutive digits or syntactically incomplete last tool call; diagnostic'),
        diagnostics=diagnostics, configurations=summarize(groups),
        sensitivity_exactly16=summarize(clean16), sensitivity_exactly16_labels_agree=summarize(label_clean))
    return report, breakdowns, flat, select_audit(flat)



def summarize_audit(notes, flat):
    by_id = {r['id']: r for r in flat}
    ids = [entry['id'] for entry in notes['entries']]
    if len(set(ids)) != len(ids) or any(i not in by_id for i in ids):
        raise ValueError('duplicate or unavailable audit IDs')
    reviewed = [dict(by_id[entry['id']], review=entry) for entry in notes['entries']]
    return dict(review_type=notes['review_type'], limitations=notes['limitations'],
                responses=len(reviewed), strata=dict(Counter(e['stratum'] for e in notes['entries'])),
                judgments=dict(Counter(e['judgment'] for e in notes['entries']))), reviewed


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recovered',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--audit-notes',type=Path,help='optional reviewed IDs and qualitative judgments')
    args = parser.parse_args()
    records = [json.loads(line) for line in args.recovered.read_text(encoding='utf-8').splitlines()]
    report, breakdowns, flat, audit = compare(records)
    if args.audit_notes:
        notes = json.loads(args.audit_notes.read_text(encoding='utf-8'))
        report['audit_review'], reviewed = summarize_audit(notes, flat)
    else:
        reviewed = []
    report['recovered_sha256'] = hashlib.sha256(args.recovered.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(args.output_dir/'summary.csv',[dict(configuration=k,**v) for k,v in report['configurations'].items()])
    write_csv(args.output_dir/'breakdowns.csv',breakdowns)
    for name, rows in [('scored_responses.jsonl',flat),('audit_examples.jsonl',audit),('reviewed_examples.jsonl',reviewed)]:
        with (args.output_dir/name).open('x',encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(report['diagnostics'],indent=2,allow_nan=False))
    for name, metrics in report['configurations'].items():
        print(name, json.dumps({k: metrics[k] for k in ('nonzero_shaping_rate','mean_bonus','incorrect_partial_credit_rate','all_wrong_gaining_variance','zero_advantage_group_rate','bonus_correctness_correlation')}))


if __name__=='__main__':
    main()
