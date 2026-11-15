"""CPU-only regression tests for on-policy distillation. Requires torch."""
import math
import pytest

torch = pytest.importorskip("torch")
from part5_distillation import distillation_loss, validate_distill_weight


def test_identical_policies_give_zero_signal():
    student = torch.tensor([[0.2, 0.4]], requires_grad=True)
    loss, kl, count = distillation_loss(student, student.detach(), torch.tensor([[1, 2]]))
    assert count == 2 and abs(kl.item()) < 1e-8 and abs(loss.item()) < 1e-8
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_teacher_is_frozen_and_sign_points_towards_teacher():
    # Student log p = -2; teacher log q = -1 => reward = +1;
    # gradient descent should INCREASE student log p (decrease student NLL).
    student_nll = torch.tensor([[2.0]], requires_grad=True)
    teacher_nll = torch.tensor([[1.0]], requires_grad=True)
    loss, kl, count = distillation_loss(student_nll, teacher_nll, torch.tensor([[9]]))
    assert count == 1 and math.isclose(kl.item(), -1.0)
    loss.backward()
    assert student_nll.grad.item() > 0   # negative GD update decreases NLL
    assert teacher_nll.grad is None


def test_masked_tokens_have_no_gradient():
    student = torch.tensor([[1.0, 10.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 0.0, 3.0]])
    targets = torch.tensor([[10, -1, 3]])
    loss, kl, count = distillation_loss(student, teacher, targets)
    assert count == 2
    loss.backward()
    assert student.grad[0, 1] == 0


def test_fully_masked_stays_finite():
    student = torch.tensor([[10.0, 20.0]], requires_grad=True)
    loss, kl, count = distillation_loss(student, torch.zeros_like(student), torch.full((1, 2), -1))
    assert count == 0 and torch.isfinite(loss) and kl.item() == 0
    loss.backward()
    assert torch.all(student.grad == 0)


@pytest.mark.parametrize("weight", [0.0, -0.2, float("nan"), float("inf")])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):
        validate_distill_weight(weight)


def test_shapes_must_align():
    with pytest.raises(ValueError):
        distillation_loss(torch.zeros(2, 3), torch.zeros(2, 4), torch.ones(2, 3, dtype=torch.long))
