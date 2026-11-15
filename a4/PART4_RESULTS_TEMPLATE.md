# CSC490 A4 Part 4 — Results and Analysis Template

> **Pre-run template; no new RL training or test results exist yet.** Fill measured values only after the completed Modal jobs and paired GSM8K evaluation. The Method B used in the initial runs is **B1 strict reference-aligned arithmetic**, not B2.

## Experimental motivation (Part 3 evidence)

The original correctness-only RL experiment increased paired greedy GSM8K accuracy from **89/1319 (6.75%)** for SFT to **194/1319 (14.71%)** for RL. These are historical results and should be reconfirmed in the paired evaluation where feasible. Part 3 found **33%** of incorrect sampled training answers lacked a final answer, motivating Method A (+0.15 valid final format). Among the **7,472** original training groups, **2,815** had no correct final answer in 16 sampled responses, motivating Method B (up to +0.20 for verified reference-aligned intermediate arithmetic).

These error labels are exploratory heuristics, not verified causes. The original RL run restarted optimizer state after an interruption, so the historical baseline is not an uninterrupted compute-matched control.

## Fixed model and training controls

- SFT start: `chatsft_checkpoints/a4-sft-default/model_000466.pt` and matching `meta_000466.json`; same tokenizer and checkpoint bytes for every run.
- Workspace: `royce-csc490`, volume `a4-base`. Confirm both on the CLI and dashboard before **any** remote launch.
- Config: `a4/part4_experiments.json`, using the pinned nanochat revision. Preserve original sampling, optimizer and epoch configuration.
- Method A: `R_A=0.15 I[valid completed #### number]`.
- Method B1: `R_B=0.20 (# unique verified exact reference-aligned steps)/(# eligible unique reference steps)`; zero when no eligible reference steps.
- Training reward: correctness + enabled bonus(es). **Evaluation always uses original binary final-answer correctness only.**
- Training runs: `a4-p4-completion-v1`, `a4-p4-arithmetic-v1`, `a4-p4-combined-v1`. A fresh `a4-p4-baseline-v1` is an optional improved control if credits permit.

## Execution and artifacts (fill after runs)

| Run | Starting SFT checkpoint hash | Final RL checkpoint/step | W&B URL | Status |
|---|---|---|---|---|
| Historical correctness baseline | TODO | `a4-rl3-resume` / 466 | TODO | historical |
| Optional uninterrupted baseline | TODO | TODO | TODO | not run |
| Completion (A) | TODO | TODO | TODO | not run |
| Arithmetic (B1) | TODO | TODO | TODO | not run |
| Combined (A+B1) | TODO | TODO | TODO | not run |

Record exact `model_*.pt` and `meta_*.json` filenames, hashes, Modal cost and completion marker. Do not evaluate incomplete or stale checkpoints as final.

## Paired GSM8K main/test results (1,319 questions, greedy, 512 tokens)

| Model | Correct | Accuracy | Delta from baseline (percentage points) | Paired 95% CI | Invalid/missing final answer |
|---|---:|---:|---:|---|---:|
| SFT starting model | 89 historical | 6.75% historical | — | TODO | TODO |
| Original RL baseline | 194 historical | 14.71% historical | 0 | TODO | TODO |
| Completion | TODO | TODO | TODO | TODO | TODO |
| Arithmetic B1 | TODO | TODO | TODO | TODO | TODO |
| Combined A+B1 | TODO | TODO | TODO | TODO | TODO |
| Optional fresh baseline | TODO | TODO | TODO | TODO | TODO |

Use identical test-question indices and answer extraction for all runs; report bootstrap CIs from paired question resampling. Do not compare the 26% sampled train-rollout reward to greedy test accuracy.

## Required plots and diagnostics

1. Mean **raw correctness** and total shaped training reward by optimizer step (distinct axes/curves, do not mislabel bonuses as accuracy).
2. GSM8K test accuracy for each final checkpoint, with paired uncertainty.
3. Failure breakdown: missing/malformed `####`, incorrect final numeric answer, and cautious qualitative operation/quantity errors.
4. Bonus distribution, proportion of zero-advantage groups, and matched intermediate steps for A/B1/A+B1.
5. Question length and numeric-quantity complexity stratification. Treat subgroup comparisons as exploratory.

## Interpretation checklist

- Did A reduce the missing-final-answer rate without increasing premature or arbitrary `####` outputs?
- Did B1 improve final-answer accuracy, rather than only increasing reference-step matches?
- Did A+B1 improve beyond A and B1 individually?
- Do results hold when comparing to the optional uninterrupted correctness-only control?
- Are semantic quantity/unit mismatches, alternative valid solution paths, or reward hacking present?
- Are changes larger than paired uncertainty and potential between-run stochastic variation?

## Limitations to report

Historical RL resumed without optimizer restoration. Limited training seeds, a small FoX model, reference-path dependence of B1, imperfect semantic grounding, GSM8K annotations that do not cover every valid step, heuristic failure categories, and mixed decoding conditions between training rollouts and greedy evaluation. The recovered 939-group offline analysis only established reward *signal availability*, not a training gain.

See `PART4_RUNBOOK.md` for workspace/billing verification and future manual commands. **No GPU run is authorized by this template.**
