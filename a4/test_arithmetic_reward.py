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
    # The parser deliberately skips expressions with nested parentheses in
    # the model extraction for now, but does support the first two equations.
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
