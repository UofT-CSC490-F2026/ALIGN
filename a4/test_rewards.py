
import pytest

from rewards import completion_reward, has_valid_final_answer


@pytest.mark.parametrize(
    "response",
    [
        "5 + 3 = 8\n#### 8",
        "The total is 16.\n#### 16",
        "Calculation complete.\n#### -12",
        "Calculation complete.\n#### 1,250",
        "Calculation complete.\n#### 2.5",
    ],
)
def test_valid_answers(response):
    assert has_valid_final_answer(response)
    assert completion_reward(response) == pytest.approx(0.15)


@pytest.mark.parametrize(
    "response",
    [
        "The answer is 8.",
        "5 + 3 = 8",
        "####",
        "#### eight",
        "#### 8\nMore reasoning afterwards",
        "#### 8\n#### 9",
        "#### 8 apples",
        "",
    ],
)
def test_invalid_answers(response):
    assert not has_valid_final_answer(response)
    assert completion_reward(response) == 0.0


def test_configurable_weight():
    assert completion_reward("#### 8", weight=0.25) == 0.25


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        completion_reward("#### 8", weight=-0.1)
