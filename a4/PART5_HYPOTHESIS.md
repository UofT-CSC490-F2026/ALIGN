# CSC490 A4 Part 5 — pre-experiment hypothesis and execution plan

**Status:** Hypothesis recorded before any Part 5 training experiment. This commit is not a claim of performance improvement; no Part 5 GPU run has been launched.

## Chosen method: on-policy log-probability distillation (fixed teacher)

We will implement a variant of Thinking Machines Lab's on-policy distillation on top of our existing nanochat GSM8K RL loop (see https://thinkingmachines.ai/blog/on-policy-distillation/). The **student** starts from the SFT checkpoint `a4-sft-default/model_000466.pt`; it generates its *own* responses to GSM8K train questions. A **fixed, frozen teacher** (initial candidate: the completed `a4-p4-completion-v1/model_000466.pt`, subject to a matched validation comparison and checkpoint verification) evaluates those same student tokens via teacher-forced next-token log probabilities. We update only the student, with an on-policy reverse-KL estimator based on the per-token difference between student and teacher log probabilities. This is distinct from simply running SFT on teacher-generated answers, and distinct from the scalar correctness-based RL rewards used in Part 4.

**Hypothesis (committed before experiments):** Compared with the unmodified SFT initialization and correctness-only RL at a matched training/evaluation budget, token-level guidance from a fixed teacher on *student-sampled trajectories* will provide denser credit assignment for partially incorrect GSM8K solutions and may improve greedy GSM8K test accuracy or reach comparable accuracy in fewer optimizer updates. Because the candidate teacher is itself a small model, we do **not** assume improvement; insufficient teacher quality may limit or reverse gains. We will test whether improvements, if any, survive ablations and full held-out evaluation.

The on-policy objective uses the student's sampled tokens `x_t`, student log-probabilities `log p_theta(x_t | x_<t)`, and teacher log-probabilities `log q(x_t | x_<t)`. For valid generated tokens, use detached per-token advantages `A_t = -(log p_theta - log q)` in the score-function/REINFORCE-style loss; stop gradients through the teacher and through the advantage. Mask prompt, padding, tool-inserted or otherwise non-trainable tokens and normalize by valid token count. This is an estimator motivated by reverse KL, not necessarily an exact full-vocabulary KL calculation. The final greedy GSM8K evaluation uses binary correctness, not KL or shaped reward.

## Implementation checklist

1. Build **separate** `a4/part5_distillation.py` pure helpers for teacher/student token alignment, masking, advantage computation, and loss normalization; unit-test zero signal for identical policies, no teacher gradient, mask behavior, finite values, and correct sign. Avoid importing Modal into pure tests.
2. Add dedicated `a4/nanochat/scripts/chat_distill.py` rather than modifying the Part 4 RL training loop. Load the student at SFT step 466 and a frozen teacher from `/vol/chatrl_checkpoints/`; verify matching tokenizer/config and teacher `COMPLETE` marker. Reuse GSM8K split, student rollout generation, distributed startup, optimizer/lr settings, W&B logging, volume checkpoints, and run IDs. Never use the test split to train or choose per-example targets.
3. Wire a `distill` command into the existing `launcher_config.py`/`modal_launch.py`, with explicit teacher tag and student checkpoint, one-step and bounded smoke modes, dedicated output paths and collision guards. Do not change the active Part 4 training behavior.
4. Log train loss, tokenwise KL-estimator statistics (careful that individual samples may be negative), sampled correctness, valid tokens, sequence lengths, GPU runtime, and evaluation curves in W&B; retain metadata/config and final `COMPLETE` marker.
5. Validate syntax and CPU tests locally; then run one tightly bounded Modal smoke test (billable, separately authorized) before the main distillation run. Confirm the teacher/student fit and measure throughput/credit use before committing to longer training.
6. Evaluate with paired greedy GSM8K test (1,319 questions, up to 512 output tokens) after training and compare with SFT and Part 3/4 models on the same decoding settings.

## Proposed ablations (same seed, data order and update budget wherever feasible)

- **Primary:** fixed teacher + on-policy per-token reverse-KL signal.
- **Ablation A:** shorter student rollouts (e.g. 128 vs 256 tokens) to study token/cost efficiency.
- **Ablation B:** alternate teacher checkpoint (e.g. arithmetic B1 vs completion A) if matched validation identifies a plausible stronger/different teacher.
- **Ablation C (budget-permitting):** off-policy teacher-response SFT with the same questions/update budget, to isolate on-policy benefit. This is not equivalent to on-policy distillation.

Finalize the ablation budget **before** training; do not use held-out GSM8K test scores to choose configurations.

## Cost and design safeguards

Part 5 is worth 20 marks and explicitly requires a *hypothesis committed before running*, one distillation run logged to W&B, a few ablations, and a comparison with RL. The existing `$500` Modal credit budget is shared with Part 4; compute limits are not billing caps. Reuse saved teacher checkpoints in `royce-csc490/a4-base` rather than invoking a paid teacher API. Do not launch Part 5 training until implementation, CPU tests, teacher checkpoint verification and a billable-run decision are complete.

## References

- Kevin Lu / Thinking Machines Lab (2025), *On-Policy Distillation*, https://thinkingmachines.ai/blog/on-policy-distillation/
- Jonas Hübotter et al. (2026), *Reinforcement Learning via Self-Distillation*, https://arxiv.org/abs/2601.20802 — alternate feedback-conditioned self-teacher method; not the method selected for this initial implementation.
