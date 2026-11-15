# Part 5 trainer (PR #45; stacked on hypothesis PR #44)

**Not yet GPU-validated. No Part 5 job has been launched.** This is implementation code for a fixed-teacher on-policy reverse-KL experiment, not an experimental result.

## Training objective

Student starts at SFT step **466** and samples GSM8K train-set responses using its own policy. A **frozen completed RL teacher** scores the *same generated tokens* using teacher-forced negative log likelihood. For each valid response token:
`A_t = stopgrad(logp_teacher - logp_student)`.
The policy-gradient estimator minimizes `-mean(logp_student * A_t)` with prompt, padding and forced tokens masked. This estimates the gradient of per-token reverse KL; individual sampled differences (and displayed estimates) can be negative, so they should not be interpreted as exact KL values.

The correctness reward from the original RL script is calculated only for diagnostics. It is NOT part of the Part 5 loss. Teacher and student must have identical architecture and tokenizer; teacher `COMPLETE` marker required. Both checkpoints live in Royce's personal `a4-base` volume.

The main trainer is `a4/nanochat/scripts/chat_distill.py`; the isolated tensor objective is `a4/part5_distillation.py`. Modal `--script distill` supports the existing `dry`, `smoke`, `full` modes, with **one update** for dry/smoke, **10 minutes** for eight-H100 smoke, and **four hours** for eight-H100 full. Run names are isolated; existing outputs cannot be overwritten. W&B project is `nanochat-distill`.

## Local preflight (free)

From repository root on branch `royce/a4-part5-code`:

```powershell
py -3.12 -m pytest a4/test_part5_launcher.py a4/test_part5_distillation.py a4/test_rewards.py a4/test_arithmetic_reward.py a4/test_reward_integration.py a4/test_modal_launch.py -q
py -3.12 -m compileall -q a4
py -m modal profile current
py -m modal volume ls a4-base /chatrl_checkpoints/a4-p4-completion-v1
py -m modal volume ls a4-base /chatsft_checkpoints/a4-sft-default
```

`test_part5_distillation.py` requires local PyTorch and will explicitly skip if it is missing; **a skipped test is not a full local validation**. Verify that `a4-p4-completion-v1` contains a `COMPLETE` marker and `model_000466.pt`.

## Suggested experiment identifiers

- Main: `a4-p5-distill-completion-v1`, teacher `a4-p4-completion-v1`.
- Shorter trajectory ablation: `a4-p5-distill-short128-v1`, same teacher, `--max-new-tokens 128`.
- Alternative teacher ablation: `a4-p5-distill-arithmetic-v1`, teacher `a4-p4-arithmetic-v1`.

Keep optimizer updates and starting checkpoint identical for a controlled ablation. Select the update budget only after benchmarking teacher + student GPU memory and throughput; a 467-step distillation run might take longer than Part 4 RL and exhaust credits. The `--max-steps N` override is accepted only in `full` mode.

## Planned smoke command (DO NOT execute before review/authorization)

```powershell
py -m modal run a4/modal_launch.py --script distill --mode smoke --run a4-p5-distill-completion-v1-smoke --start a4-sft-default --teacher-tag a4-p4-completion-v1 --extra "--num-epochs 1 --device-batch-size 8 --examples-per-step 16 --num-samples 16 --max-new-tokens 256 --temperature 1 --top-k 50 --model-step 466"
```

This is **billable** despite being a smoke test. Check Modal credits and obtain explicit approval before launching. The smoke test is particularly important here because each GPU holds *two models* and performs a teacher forward pass.

## Post-training evaluation

The output checkpoints are in `/vol/chatrl_checkpoints/<run>`, so existing evaluation syntax can refer to them as `rl:<run>` (they are distillation-trained weights, not an RL method). Compare against SFT, correctness-only RL, Methods A/B1/combined using matched 1,319-question greedy GSM8K test evaluation and the same decoder/token budget. For teacher checkpoints and Part 4 results, avoid modifying their original directories.
