"""Synthetic local rollout tests, including malformed and overlapping input."""
import json
import math
from pathlib import Path
import pytest
from analyze_reward_signals import analyze, save_summary, summarize


def record(problem=1):
    return dict(step=0, problem=problem, question='How much?', reference='<<3*4=12>>\n#### 12', samples=[
        dict(text='3*4=12\n#### 13', reward=0),
        dict(text='#### 12', reward=1),
        dict(text='unfinished', reward=0),
    ])


def write(path, rows):
    path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')


def test_signals_variance_and_correctness_separate(tmp_path):
    p=tmp_path/'input.jsonl'
    write(p, [record()])
    report=analyze([p])
    result=report['all']
    assert result['responses']==3
    assert result['correctness_rate']==pytest.approx(1/3)
    assert result['reference_step_coverage']==1
    configs=result['configurations']
    assert configs['combined']['mean_reward']==pytest.approx((.35+1.15)/3)
    assert configs['arithmetic']['incorrect_partial_credit_rate']==.5
    assert configs['completion']['positive_reward_rate']==pytest.approx(2/3)
    assert configs['baseline']['mean_reward']==pytest.approx(1/3)
    for metrics in configs.values():
        assert metrics['mean_within_question_reward_variance']==metrics['mean_within_question_advantage_variance']
    assert configs['combined']['mean_within_question_reward_variance']>0
    output=tmp_path/'out'
    save_summary(report, output)
    assert {p.name for p in output.iterdir()}=={'summary.csv','summary.json','positive_rates.svg'}
    assert json.loads((output/'summary.json').read_text())['all']==result
    with pytest.raises(FileExistsError):
        save_summary(report, output)


def test_overlaps_malformed_lines_and_samples(tmp_path):
    first=tmp_path/'first.jsonl'; second=tmp_path/'second.jsonl'
    row=record(); conflict=record(); conflict['question']='different'
    invalid=record(2); invalid['samples'] += [dict(text='x', reward=.7), None]
    write(first,[row,invalid,{'step':'bad'}])
    with first.open('a') as f: f.write('\nnot json')
    write(second,[row,conflict])
    report=analyze([first,second])
    assert report['input_counts']==dict(lines=6,duplicate_records=1,conflicting_records=1,
                                       malformed_lines=1,malformed_records=1,malformed_samples=2)
    assert report['all']['records']==2
    assert report['all']['responses']==6
    assert len(report['issues'])==6


def test_missing_annotations_and_constant_labels(tmp_path):
    row=record(); row['reference']='#### 12'; row['samples']=[dict(text='#### 12',reward=1)]
    p=tmp_path/'in.jsonl'; write(p,[row])
    report=analyze([p])['all']
    assert report['mean_arithmetic_bonus']==0
    assert report['configurations']['baseline']['reward_correctness_correlation'] is None
    assert report['configurations']['baseline']['incorrect_partial_credit_rate'] is None


def test_new_schema_keeps_correctness_over_shaped_total(tmp_path):
    row=record(); row['samples']=[dict(text='#### 13',correctness=0,reward=.15,total=.15)]
    p=tmp_path/'in.jsonl'; write(p,[row])
    assert analyze([p])['all']['correctness_rate']==0


def test_empty_and_invalid_weights():
    result=summarize([])
    assert result['responses']==0
    assert result['configurations']['baseline']['mean_reward'] is None
    for a,b in [(float('nan'),.2),(.15,float('inf')),(-1,.2),(.15,.3)]:
        with pytest.raises(ValueError): summarize([],a,b)
