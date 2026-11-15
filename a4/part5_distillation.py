"""Pure tensor helpers for CSC490 A4 Part 5 on-policy distillation.

The student samples on-policy tokens. On those tokens, a frozen teacher supplies
log q and the current student supplies log p. With a zero-discount per-token
reverse-KL signal, use the detached advantage log q - log p in the REINFORCE
loss. This is a sampled gradient estimator, NOT an exact nonnegative KL value.
"""
import math


def validate_distill_weight(weight):
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("distillation weight must be finite and positive")


def distillation_loss(student_nll, teacher_nll, targets, weight=1.0):
    """Return (policy-gradient loss, sampled reverse-KL estimate, valid token count).

    Each NLL is [batch, sequence] and must represent aligned token targets.
    targets == -1 marks prompt/padding/forced tokens that must not contribute.
    The teacher must be evaluated without gradients. The advantage is detached:
    optimizing the sampled log-prob with its own non-detached advantage would
    introduce an incorrect extra gradient term.
    """
    import torch

    validate_distill_weight(weight)
    if student_nll.shape != teacher_nll.shape or student_nll.shape != targets.shape:
        raise ValueError("student, teacher and targets must have matching shapes")
    valid = targets.ne(-1)
    n_valid = valid.sum().clamp(min=1)
    logp = -student_nll
    logq = -teacher_nll.detach()
    advantage = (logq - logp.detach()) * valid
    loss = -(logp * advantage).masked_fill(~valid, 0).sum() / n_valid
    approx_reverse_kl = ((logp.detach() - logq) * valid).sum() / n_valid
    return weight * loss, approx_reverse_kl.detach(), int(valid.sum().item())
