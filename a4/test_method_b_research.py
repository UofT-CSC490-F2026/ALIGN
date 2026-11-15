"""CPU-only tests for reference recovery and diagnostic Method B candidates."""
import ast
import copy
import json
from fractions import Fraction
from pathlib import Path
import pytest
from arithmetic_reward import completion_calculations
from recover_gsm8k_references import normalize_question, recover, fetch_train, TRAIN_URL
from method_b_candidates import (candidate_signals, commutative_key, canonical_steps,
                                 grounded_matches, question_quantities)
from compare_method_b import compare, varying, safe_failure, summarize_audit
from part3_failure_labels import failure_kind


def tool(expression,result):
    return f'<|python_start|>{expression}<|python_end|><|output_start|>{result}<|output_end|>'


def score(reference,text,question='There are 3 boxes with 4 items each.'):
    return candidate_signals(question,reference,text)


def record(problem=0,reference='tail\n#### 12',question='Three boxes?',samples=None):
    return dict(step=0,problem=problem,source='synthetic',question=question,reference=reference,
                samples=samples or [dict(text='#### 13',correctness=0)])


def test_recovery_exact_normalization_and_shuffle_validation():
    train=[dict(question='Three boxes?',answer='<<3*4=12>>\n#### 12'),
           dict(question='Other?',answer='<<2+2=4>>\n#### 4')]
    rows=[record(1,question='  Three\n boxes? '),record(0,reference='#### 4',question='Other?')]
    recovered,report=recover(rows,train,[1,0])
    assert report['counts']['matched']==report['counts']['index_verified']==2
    assert report['counts']['valid_annotation_coverage']==2
    assert recovered[0]['historical_reference']=='tail\n#### 12'
    assert rows[0]['reference']=='tail\n#### 12'
    assert recovered[0]['reference']=='<<3*4=12>>\n#### 12'
    assert normalize_question('Ａ  B')=='A B'
    assert normalize_question('Case?')!=normalize_question('case?')


def test_recovery_unmatched_ambiguous_bad_index_and_final():
    train=[dict(question='dup',answer='#### 1'),dict(question='dup',answer='#### 2'),
           dict(question='Three boxes?',answer='<<3*4=12>>\n#### 12')]
    rows=[record(0,question='dup'),record(1,question='absent'),record(0,reference='#### 99')]
    recovered,report=recover(rows,train,[0,1,2])
    assert len(recovered)==1
    for key in ('ambiguous','unmatched','index_mismatch','historical_final_mismatch'):
        assert report['counts'][key]==1
    assert not recovered[0]['recovery']['historical_final_matches']


def test_train_fetch_no_overwrite_and_train_only(tmp_path,monkeypatch):
    p=tmp_path/'train.jsonl'; p.write_text('keep')
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:pytest.fail('unexpected network access'))
    with pytest.raises(FileExistsError): fetch_train(p)
    assert p.read_text()=='keep'
    assert TRAIN_URL.endswith('/data/train.jsonl') and '/test' not in TRAIN_URL


@pytest.mark.parametrize('expression,result', [('4*3','12'),('4+3','7'),('(4*3)+2','14'),('2*(4+3)','14')])
def test_commutative_equivalence(expression,result):
    reference={'12':'<<3*4=12>>','7':'<<3+4=7>>','14':'<<(3*4)+2=14>>' if expression.startswith('(') else '<<2*(3+4)=14>>'}[result]
    values=score(reference,tool(expression,result))
    assert values['B2']==.2
    assert values['B1']==0


@pytest.mark.parametrize('reference,expression,result',[
    ('<<9-3=6>>','3-9','-6'),('<<12/3=4>>','3/12','.25'),
    ('<<(1+2)+3=6>>','1+(2+3)','6'),('<<3*4=12>>','6*2','12'),
])
def test_no_noncommutative_or_algebraic_expansion(reference,expression,result):
    assert score(reference,tool(expression,result))['B2']==0


@pytest.mark.parametrize('text', [tool('3*4','13'),tool('print(3*4)','12'),
    tool('__import__("os").system("anything")','12'),tool('12','12'),
    tool('2**999999','12'),tool('[3*4][0]','12'),'12 3 4',tool('6*2','12')])
def test_no_arbitrary_math_numbers_or_unsafe_code(text):
    result=score('<<3*4=12>>',text+'\n#### 12')
    assert result['B1']==result['B2']==result['B3']==result['B4']==0


def test_repeated_and_reordered_steps_count_once():
    reference='<<3*4=12>> <<4*3=12>> <<12+2=14>>'
    result=score(reference,(tool('4*3','12')+'\n')*20)
    assert result['B2']==pytest.approx(.1)
    assert result['B1']==pytest.approx(.2/3)


def test_grounding_dependency_closure_and_limitations():
    reference='<<3*4=12>> <<12+5=17>>'
    result=score(reference,tool('5+12','17')+tool('4*3','12'),question='3 boxes of 4 items and 5 extras')
    assert result['B2']==result['B3']==.2
    # A valid reference step with an unstated unit conversion constant is rejected.
    result=score('<<3*52=156>>',tool('3*52','156'),question='3 pages per week for a year')
    assert result['B2']==.2 and result['B3']==0
    # Numeric equality cannot tell whether the right entity/unit was used.
    result=score('<<3*4=12>>',tool('3*4','12'),question='3 children with 4 apples; also 4 cars')
    assert result['B3']==.2
    assert Fraction(12) in question_quantities('a dozen eggs')


def test_consistency_requires_valid_relevant_last_tool():
    reference='<<3*4=12>> <<12+5=17>>\n#### 17'
    # Reports an earlier reference step: internally consistent but WRONG final.
    assert score(reference,tool('3*4','12')+'\n#### 12')['B4']==.05
    unrelated=score(reference,tool('6*2','12')+'\n#### 12')
    assert unrelated['B4']==0 and unrelated['diagnostics']['ungated_consistency']
    text=tool('3*4','12')+tool('2+2','5')+'\n#### 12'
    assert score(reference,text)['B4']==0
    assert score(reference,tool('3*4','12')+'<|python_start|>2+2\n#### 12')['B4']==0
    assert score(reference,tool('3*4','12')+'\n#### 12\n#### 17')['B4']==0


def test_penalty_is_bounded_negative_and_cause_unknown():
    for text in ['9'*12,'<|python_start|>3*4',tool('3*4','12')+'<|python_start|>1+', '9'*12+'<|python_start|>1+']:
        result=score('<<3*4=12>>',text)
        assert result['B5']==-.05
        assert result['diagnostics']['termination_cause']=='unknown'
    assert score('<<3*4=12>>',tool('3*4','13'))['B5']==0
    assert score('<<3*4=12>>','completed prose.')['B5']==0


def test_signal_metrics_all_wrong_variance_and_A_increment():
    rows=[record(samples=[dict(text=tool('3*4','12')+'\n#### 13',correctness=0),
                          dict(text='#### 15',correctness=0)])]
    recovered,_=recover(rows,[dict(question='Three boxes?',answer='<<3*4=12>>\n#### 12')],[0])
    report,breakdowns,flat,audit=compare(recovered)
    assert report['configurations']['baseline']['all_wrong_groups']==1
    assert report['configurations']['A']['all_wrong_gaining_variance']==0
    assert report['configurations']['B1']['all_wrong_gaining_variance']==1
    assert report['configurations']['A+B1']['additional_all_wrong_differentiated_beyond_A']==1
    assert report['configurations']['B1']['incorrect_partial_credit_rate']==.5
    assert report['configurations']['B1']['mean_bonus']==.1
    assert report['configurations']['B1']['bonus_correctness_correlation'] is None
    assert {r['dimension'] for r in breakdowns}=={'step_bin','length_bin','quantity_bin','failure','step'}
    assert all(r['correct']==0 for r in flat)
    assert not varying([.2,.2+1e-16])


def test_empty_and_unverified_inputs():
    report,breakdowns,flat,audit=compare([record()])
    assert report['diagnostics']['groups']==0 and not flat and not audit
    assert report['configurations']['B1']['responses']==0


def test_heuristic_copy_matches_original_without_heavy_imports():
    import re
    from part3_failure_labels import FINAL,NUMBER,TOOL_RESULT,PYTHON,REPEAT,COMMON,number,final_answer
    tree=ast.parse(Path('a4/analyze_rl.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='failure_kind')
    namespace=dict(number=number,final_answer=final_answer,FINAL=FINAL,NUMBER=NUMBER,
                   TOOL_RESULT=TOOL_RESULT,PYTHON=PYTHON,REPEAT=REPEAT,COMMON=COMMON)
    # This is trusted repository source, never generated arithmetic/output.
    exec(compile(ast.Module(body=[copy.deepcopy(function)],type_ignores=[]),'original','exec'),namespace)
    for text in ['unfinished', '1'*15, '#### 4', '#### 6', '#### 12', tool('2*3','6')+'\n#### 20']:
        assert failure_kind('2 boxes with 3 each','#### 6',text)==namespace['failure_kind']('2 boxes with 3 each','#### 6',text)



def test_quantity_literals_followed_by_units_or_sentence_punctuation():
    assert {Fraction(210),Fraction(130),Fraction('6.80')} <= question_quantities('210kg then 130kg; socks cost $6.80.')



def test_audit_annotations_are_validated_and_do_not_change_scores():
    rows=[record(samples=[dict(text=tool('3*4','12')+'\n#### 13',correctness=0)])]
    recovered,_=recover(rows,[dict(question='Three boxes?',answer='<<3*4=12>>\n#### 12')],[0])
    report,_,flat,_=compare(recovered)
    original=copy.deepcopy(flat)
    notes=dict(review_type='test',limitations='test only',entries=[dict(id='0:0:0',stratum='credited',judgment='partial')])
    summary,reviewed=summarize_audit(notes,flat)
    assert summary['responses']==1 and summary['judgments']=={'partial':1}
    assert flat==original and reviewed[0]['review']['id']=='0:0:0'
    for ids in ([notes['entries'][0]]*2,[dict(id='absent')]):
        with pytest.raises(ValueError): summarize_audit(dict(notes,entries=ids),flat)


def test_semantic_aliasing_is_a_documented_risk_not_a_proof():
    reference='<<4*30=120>> <<30*8=240>> <<240/120=2>>'
    text='There are 30*4='+tool('30*4','120')+' buns.\n#### 120'
    result=score(reference,text,question='4 classes of 30 students and 30 packages of 8 buns')
    assert result['B2']>0  # Reference student count is mislabeled as buns.
    assert result['B4']==.05  # Last-result agreement would reinforce the same mistake.


def test_word_or_percent_quantity_coverage_is_not_semantic_grounding():
    correct=score('<<500*.8=400>>',tool('500*.8','400'),question='80 percent of 500')
    assert correct['B2']==.2 and correct['B3']==0
    assert correct['diagnostics']['ungrounded_operand_values']>0


def test_normal_floating_output_can_trigger_repetition_penalty():
    result=score('<<100*1.1=110>>',tool('100*1.1','110.00000000000001'))
    assert result['B1']==.2
    assert result['B5']==-.05


def test_audit_file_contains_at_least_20_credited_and_20_uncredited_reviews():
    notes=json.loads(Path('a4/method_b_audit.json').read_text())
    strata=[entry['stratum'] for entry in notes['entries']]
    assert sum(s in ('B1_credit_wrong','B2_only_credit') for s in strata)>=20
    assert sum(s in ('B2_no_credit_with_tools','B2_no_credit_no_tools') for s in strata)>=20
    assert len({e['id'] for e in notes['entries']})==len(notes['entries'])
