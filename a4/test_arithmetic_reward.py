import pytest

from arithmetic_reward import (
    arithmetic_reward,
    arithmetic_score,
    completion_calculations,
    reference_calculations,
)

REFERENCE = (
    "Notebook cost: <<3*4=12>>12 dollars. "
    "Pens cost: <<2*2=4>>4 dollars. "
    "Total: <<12+4=16>>16. #### 16"
)


def test_full_match():
    assert arithmetic_score(REFERENCE, "3 * 4 = 12\n2*2=4\n12+4=16\n#### 16") == 1.0
    assert arithmetic_reward(REFERENCE, "3*4=12\n2*2=4\n12+4=16") == pytest.approx(0.20)


def test_partial_credit_even_with_wrong_final():
    text = "3 * 4 = 12\n2 * 2 = 6\n12 + 6 = 18\n#### 18"
    assert arithmetic_score(REFERENCE, text) == pytest.approx(1 / 3)
    assert arithmetic_reward(REFERENCE, text) == pytest.approx(0.20 / 3)


def test_numbers_alone_do_not_earn_credit():
    assert arithmetic_reward(REFERENCE, "12, 4, 16. #### 16") == 0


def test_incorrect_equalities_do_not_earn_credit():
    assert arithmetic_reward(REFERENCE, "3*4=13\n2*2=5\n12+4=17") == 0


def test_unrelated_true_equations_do_not_earn_credit():
    assert arithmetic_reward(REFERENCE, "6*2=12\n1+3=4\n8+8=16") == 0


def test_repeated_steps_do_not_inflate_score():
    assert arithmetic_score(REFERENCE, "3*4=12 " * 10) == pytest.approx(1 / 3)


def test_duplicate_reference_steps_count_once():
    ref = "<<3*4=12>> and again <<3*4=12>>. <<2*2=4>>"
    assert arithmetic_score(ref, "3*4=12") == pytest.approx(0.5)


def test_no_annotations_no_bonus():
    assert arithmetic_reward("Calculate this. #### 16", "3*4=12") == 0


def test_invalid_reference_step_is_not_credited():
    assert not reference_calculations("<<3*4=13>>")
    assert arithmetic_score("<<3*4=13>>", "3*4=12") == 0


def test_division_decimals_commas_and_parentheses():
    ref = "<<1200/3=400>> <<2.5+1.5=4>> <<(2+3)*4=20>>"
    assert arithmetic_score(ref, "1,200 / 3 = 400\n2.5+1.5 = 4") == pytest.approx(2 / 3)


def test_zero_division_is_ignored():
    assert not reference_calculations("<<4/0=1>>")
    assert not completion_calculations("4/0=1")


def test_code_execution_is_impossible():
    assert arithmetic_reward("<<__import__('os').system('echo hi')=1>>", "hi") == 0


def test_empty_strings():
    assert arithmetic_score("", "") == 0


def test_weight_validation():
    with pytest.raises(ValueError):
        arithmetic_reward(REFERENCE, "3*4=12", weight=-0.1)
    with pytest.raises(ValueError):
        arithmetic_reward(REFERENCE, "3*4=12", weight=1.5)


def test_text_type_validation():
    with pytest.raises(TypeError):
        arithmetic_score(None, "3*4=12")


def tool(expression, result):
    return f'<|python_start|>{expression}<|python_end|><|output_start|>{result}<|output_end|>'


@pytest.mark.parametrize('expression,result', [
    ('(2+3)*4', '20'), ('-2 + +3', '1'), ('1,200/3', '400'),
    ('.5*8', '4.0'), ('1e2/4', '25'), ('1/3', '0.3333333333333333'),
    ('0.1+0.2', '0.30000000000000004'),
])
def test_numeric_and_tool_formats(expression, result):
    ref = f'<<{expression}={result}>>'
    assert arithmetic_score(ref, tool(expression, result)) == 1
    assert arithmetic_score(ref, f'{expression} = {result}') == 1


@pytest.mark.parametrize('expression,result', [
    ('3*4', '13'), ('6*2', '12'), ('12', '12'),
    ('__import__("os").system("echo bad")', '12'),
    ('print(3*4)', '12'), ('x=3*4', '12'), ('2**100000000', '12'),
    ('[3*4][0]', '12'), ('True*12', '12'), ('3*4 # comment', '12'),
    ('3*4', '12 extra'), ('3*4', 'nan'), ('3*4', 'inf'),
])
def test_invalid_or_unaligned_tools(expression, result):
    assert arithmetic_score('<<3*4=12>>', tool(expression, result)) == 0


def test_tool_and_ordinary_deduplicate_numeric_spellings():
    assert arithmetic_score('<<5*7=35>><<5.0*7=35.0>><<2+2=4>>',
                            tool('5*7', '35') + '\n5.0*7=35') == .5


def test_nested_parentheses_and_signs():
    assert arithmetic_score('<<((2+3)*(-4))/2=-10>>', '((2+3)*(-4))/2=-10') == 1


def test_bad_commas_and_oversized_expressions():
    assert arithmetic_score('<<12*2=24>>', '1,2*2=24') == 0
    assert not reference_calculations('<<' + '1+' * 200 + '1=201>>')


@pytest.mark.parametrize('weight', [float('nan'), float('inf'), .21])
def test_nonfinite_and_over_cap_weight(weight):
    with pytest.raises(ValueError):
        arithmetic_reward(REFERENCE, '', weight)


@pytest.mark.parametrize('prefix', ['We have ', 'There are ', 'The price is ', 'Compute ', 'Total: '])
def test_equations_following_prose(prefix):
    assert arithmetic_score('<<3*4=12>>', prefix+'3*4=12') == 1


@pytest.mark.parametrize('text', ['x + 3*4=12', 'x - 3*4=-12', '2**3*4=12', '3*4=120', '3*4=12.5'])
def test_do_not_extract_suffix_or_wrong_claim(text):
    assert arithmetic_score('<<3*4=12>>', text) == 0


def test_large_integer_output_must_not_be_off_by_one():
    assert arithmetic_score('<<1000000*1000000=1000000000000>>', tool('1000000*1000000','999999999999')) == 0


@pytest.mark.parametrize('expression', ['1e99999999+1', '1e-99999999+1', '1'*121+'+1'])
def test_extreme_numeric_literals_are_bounded(expression):
    assert not reference_calculations(f'<<{expression}=1>>')
