# Method B selection: verified intermediate arithmetic, with a narrow commutative proposal

## Recommendation and decision boundary

**B2 is the strongest narrowly scoped candidate for review; keep B1 in production
until a blinded semantic audit of B2-only credits passes. Do not add B3, B4 or B5
to the planned production reward.** B1 is already sufficient to test the core
reference-aligned partial-credit hypothesis. B2 removes an arbitrary operand-order
restriction without broadening the supported mathematics, and improves the measured
arithmetic signal modestly. This is a proposal for a controlled comparison, not
proof that B2 trains a better model or has acceptable population-level semantic
precision. No production reward, launcher mode or experiment manifest was changed
by this research. In particular, the reviewed Method A is untouched.

The main conclusion is to retain **reference-aligned verified arithmetic** as the
second reward family, rather than generic calculator correctness, numeric mention
coverage, internal answer consistency or a bundle of unrelated bonuses. Arithmetic
truth plus reference-tree equality is interpretable evidence of a useful substep;
it is still not proof that the right quantity or unit was used in context.

## Motivation from Part 3

The user-provided full Part 3 findings report 6.75% -> 14.71% GSM8K test accuracy,
33% of wrong training responses without a final answer, and 67% with an incorrect
final answer. Method A targets the first failure. Reported heuristic wrong-answer
categories include wrong setup/operation (24.4%), whole-factor errors (11.7%), omitted
quantity (11.4%), invented operand (5.7%), off-by-one/two (3.9%), reaching the right
value then reporting another (3.6%), and reporting an earlier result (2.4%). These
are mutually ordered **heuristic descriptions, not verified causes**. No causal
explanation or reward parameter is fitted to them here.

A stronger motivation than fixing calculator arithmetic is supplying partial credit
for correct task-relevant subcomputations when the final setup is still wrong.
Part 3 had 2,815/7,472 groups with 0/16 correct answers: subtracting a constant zero
reward provides no within-question learning signal. Long/quantity-rich questions
were particularly hard. Laith's implementation and labeling rules are in
`a4/analyze_rl.py`; this research copies its dependency-free heuristic functions to
`part3_failure_labels.py` and tests their equivalence on trusted repository source.
No existing analysis code is imported with torch, sklearn or plotting dependencies.

## Recovery, provenance and data quality

Raw schema: records contain `step`, `problem`, `question`, `reference`, `samples`;
samples contain only `text` and `reward`. The raw reference is the last assistant
text part, so intermediate nanochat reference tool parts were lost. The two files
contain 132 and 836 lines. They are subsets of the full 8-rank rollouts, not a random
held-out evaluation set. Raw files were opened read-only and fingerprinted.

There was no GSM8K train cache in the inspected Hugging Face cache. Downloaded only
the **4,166,206-byte / 7,473-row training JSONL** from the official
[OpenAI source](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl),
the source underlying [openai/gsm8k main](https://huggingface.co/datasets/openai/gsm8k).
This avoids installing parquet/model dependencies. No test split, model checkpoint,
paid API or compute endpoint was used. This is not a byte-for-byte assertion about
the current HF parquet artifact; the checks below verify the relevant questions,
answers and ordering for these rollouts.

Dataset SHA-256:
`17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`.

- Excluded 10 malformed JSON lines; kept first occurrences of two conflicting
  `(step, problem)` overlaps, flagging both. No fuzzy recovery or JSON repair.
- All **956/956 retained questions** uniquely match using Unicode NFKC plus whitespace
  collapse, preserving case and punctuation. No unmatched or ambiguous questions.
- All **956/956 indices** agree with `numpy.random.default_rng(42).permutation(7473)`
  (NumPy 1.26.4), reproducing the seed-42 shuffle in the
  [pinned task](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/tasks/gsm8k.py).
  Matching is by question first; the index is an independent cross-check.
- All 956 historical final reference answers agree with the recovered full answer.
- **943/956 (98.64%)** contain `<<...>>`; **936/956 (97.91%)** have at least one
  supported verified arithmetic step. Across these records there are 3,007
  annotations. The 34 unsupported annotations are bare identities/unary constants
  such as `400=400` and `+10=10`, deliberately ineligible for arithmetic credit.
  Thirteen references have no annotations; seven have only unsupported annotations.
- The full retained population has 15,285 responses. Seventeen groups have nonstandard
  sample counts (3 to 35), including apparent export contamination. Recomputing the
  pinned **string-based** final-answer correctness check disagrees with 21 original
  sample labels, all in nonstandard-sized groups. Original labels remain unchanged.
- **Primary analysis:** 939 groups with exactly 16 samples and no label disagreements
  (15,024 responses; 353 all-wrong groups). This is a data-quality sensitivity filter,
  not a selection on which reward performs best. It is not proof that every remaining
  response is undamaged. Full-retained results are separately reported below.

`recovery.json` contains input hashes, line issues and shuffle checks. Full recovered
answers and sample texts remain in the ignored output directory. No raw rollouts or
full dataset are committed.

## Candidate definitions fixed before evaluating associations

All arithmetic candidates use the existing restricted AST parser and result check;
no generated Python is executed. They reject arbitrary true equations, standalone
numbers, invalid tool outputs and repeated-step inflation. Weights were **not
optimized** against correctness or failure categories, and no GSM8K test labels
were used. B1-B3 retain the existing 0.20 cap; B4/B5 use a fixed small 0.05 diagnostic
magnitude. Reward/advantage variances and total-reward correlations depend on those
weights; a component Pearson correlation is invariant to positive rescaling.

| Candidate | Exact operational definition | Complexity/risk |
|---|---|---|
| B1 | Unmodified `arithmetic_reward(reference, text)`; fraction of unique strict reference expression/result pairs reproduced, max +0.20. | Low; order-sensitive false negatives and semantic aliases remain. |
| B2 | Sort the two recursively canonicalized children at **Add/Mult only**, then intersect unique validated reference/generated pairs; max +0.20. No reassociation, distribution, constant folding, subtraction/division swapping or approximate algebra. | Low incremental complexity; preserves numerical validity, not entity/unit identity. Canonical deduplication changes the denominator. |
| B3 | B2 matches filtered by numeric-provenance closure from question numeric values, a limited number-word lexicon and fixed structural constants `{0,1,2,100}`. Accepted reference-aligned outputs can ground other matched steps; order is not enforced. | Diagnostic only. Misses unit/percentage/word conversions; equal-valued entities are indistinguishable. |
| B4 | +0.05 only if a valid completed final answer equals the **last** verified tool result and that tool step is B2-reference-aligned. A later invalid/incomplete tool disqualifies it. | Diagnostic only. Even relevance-gated consistency can reward reporting an intermediate value as final. Ungated consistency is counted, never rewarded. |
| B5 | -0.05 once for >=12 identical consecutive digits anywhere, or a syntactically incomplete last calculator call. | Diagnostic only. Can penalize normal floating-point outputs or externally capped sequences. Not a calibrated length/repetition reward. |

A remains +0.15 for its existing completed-final-answer rule. Every total is original
binary correctness + the listed shaping component(s). Correlations below refer to
the shaping component and original correctness. A positive bonus on an incorrect
response is intended partial credit, **not automatically a false positive**. False
positives concern invalid, irrelevant or contextually misapplied substeps instead.
For B5, coverage is penalty coverage and positive partial-credit rate is zero.

## Primary learning-signal results

All 939 primary groups retain raw correctness **26.3179%**. Centered advantages
`r - mean(r)` have the same variance as rewards. A group is treated as constant when
its range is <=1e-12, avoiding floating-point noise. Group rates weight questions
equally; response coverage/means/correlations weight responses equally.

| Configuration | Nonzero shaping | Mean shaping | Incorrect positive credit | All-wrong groups differentiated /353 | Zero-advantage groups | Bonus/correctness Pearson r |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.00% | 0.000000 | 0.00% | 0 (0.00%) | 41.43% | undefined |
| A | 75.42% | 0.113129 | 66.65% | 318 (90.08%) | 7.56% | 0.3408 |
| B1 | 59.76% | 0.074579 | 47.59% | 252 (71.39%) | 13.42% | 0.7054 |
| A+B1 | 85.73% | 0.187708 | 80.63% | 339 (96.03%) | 4.15% | 0.6709 |
| B2 | 62.98% | 0.081385 | 51.48% | 258 (73.09%) | 13.10% | 0.7288 |
| A+B2 | 86.35% | 0.194514 | 81.47% | 340 (96.32%) | 4.37% | 0.6848 |
| B3 | 55.17% | 0.067958 | 45.38% | 215 (60.91%) | 17.78% | 0.6117 |
| A+B3 | 84.87% | 0.181087 | 79.47% | 335 (94.90%) | 5.01% | 0.6133 |
| B4 | 25.42% | 0.012710 | 4.53% | 74 (20.96%) | 33.23% | 0.8026 |
| A+B4 | 75.42% | 0.125839 | 66.65% | 327 (92.63%) | 6.28% | 0.5285 |
| B5 | 5.39% | -0.002696 | 0.00% | 160 (45.33%) | 24.39% | 0.1326 |
| A+B5 | 78.77% | 0.110433 | 66.65% | 323 (91.50%) | 7.03% | 0.3466 |

Against baseline's 41.43% zero-advantage rate, B1 reduces it to 13.42%, B2 to 13.10%,
B3 to 17.78%, B4 to 33.23% and B5 to 24.39%. This is descriptive signal availability,
not a causal gradient-quality or performance estimate.

A already differentiates 318/353 all-wrong groups. Among the **35 left constant by A**,
A+B1 differentiates another **21**, A+B2 **22**, A+B3 **17**, A+B4 **9** and A+B5 **5**.
Thus the incremental all-wrong gain of B2 over B1 when paired with A is **one group**.
B2's stronger arithmetic coverage/correlation does not establish a large training gain.
A+B2 has slightly *more* constant groups overall than A+B1: canonicalization also
removes spurious operand-order differences among some all-correct groups. Minimizing
zero-advantage rate indiscriminately is not the objective.

For the full 956-group retained population (360 all-wrong groups), sensitivity results:

| Candidate | Positive/negative shaping coverage | Incorrect positive credit | All-wrong differentiated | A+candidate differentiated | Bonus/correctness r |
|---|---:|---:|---:|---:|---:|
| B1 | 59.33% | 47.20% | 256/360 | 346/360 | 0.7036 |
| B2 | 62.51% | 51.03% | 262/360 | 347/360 | 0.7264 |
| B3 | 54.73% | 44.95% | 219/360 | 342/360 | 0.6100 |
| B4 | 25.25% | 4.50% | 75/360 | 331/360 | 0.8009 |
| B5 | 5.36% | 0.00% | 164/360 | 328/360 | 0.1313 |

On all retained responses, B2 gives first arithmetic credit to **485** responses
missed by B1; 1,397 have higher B2 than B1 bonuses and 13 lower. This is not a
monotone score extension: collapsing equivalent duplicate reference steps can
lower a response's fraction. Baseline/A comparisons and all A+candidate totals,
means, variances and total-reward correlations are in JSON/CSV. Results remain
training-only associations with correlated responses and no inferential p-values.

## Breakdowns: where B1/B2 provide signal

These tables use the **primary 939-group population**. Incorrect-credit columns
condition on original incorrect labels. Quantity bins count numeric mentions with
Laith's original regex, not distinct semantic entities; number words are not counted
in those bins. B3 separately uses its documented limited lexical quantity extractor.
Individual-step rows (not just stage bins), all candidates and full-retained
sensitivity rows are in `breakdowns.csv`.

| Training stage | Responses | Raw correctness | B1 incorrect credit | B2 incorrect credit |
|---|---:|---:|---:|---:|
| 0-116 | 3648 | 26.15% | 45.73% | 50.19% |
| 117-233 | 3840 | 24.01% | 43.35% | 47.70% |
| 234-350 | 3632 | 26.51% | 50.43% | 54.40% |
| 351-466 | 3904 | 28.56% | 51.09% | 53.89% |

| Question words | Responses | Raw correctness | B1 incorrect credit | B2 incorrect credit |
|---|---:|---:|---:|---:|
| <30 | 2416 | 39.03% | 37.07% | 38.56% |
| 30-49 | 7680 | 28.88% | 48.85% | 53.46% |
| 50-79 | 4032 | 17.96% | 49.43% | 53.42% |
| 80+ | 896 | 7.70% | 50.67% | 53.69% |

| Numeric mentions | Responses | Raw correctness | B1 incorrect credit | B2 incorrect credit |
|---|---:|---:|---:|---:|
| 0-1 | 1792 | 34.88% | 50.21% | 53.98% |
| 2-3 | 7728 | 29.45% | 44.33% | 48.83% |
| 4-5 | 3888 | 21.86% | 49.74% | 52.90% |
| 6+ | 1616 | 12.56% | 53.36% | 56.62% |

Correctness falls to 7.70% for 80+ word questions and 12.56% for 6+ numeric mentions,
but B1/B2 still offer partial credit to roughly half of wrong answers in those bins.
This is consistent with the desired opportunity for intermediate supervision, not
proof that omitted quantities caused the errors or that a particular bonus fixes them.

| Laith-style heuristic category | Wrong responses | B1 partial credit | B2 partial credit |
|---|---:|---:|---:|
| wrong setup or operation (all numbers present) | 2524 | 51.66% | 56.54% |
| off by a whole factor | 1423 | 53.55% | 57.41% |
| never uses a number the question gives | 1273 | 46.98% | 54.05% |
| uses a number found nowhere in the question or earlier steps | 683 | 42.90% | 44.51% |
| off by 1 or 2 | 464 | 55.17% | 58.19% |
| reaches the right value, then reports a different one | 453 | 48.34% | 52.98% |
| reports an earlier step's result | 243 | 54.73% | 61.32% |

The labels remain the original mutually ordered heuristics. They must not be used
as causal truth or as automatic semantic precision labels for a reward. A response
can contain both a useful matched prefix and a later invented quantity or omitted step.

## Qualitative precision/recall and reward-hacking audit

Reviewed **55 full responses**: 10 B1-credited wrong, 10 B2-only credited, 10 uncredited
with tools, 10 uncredited without tools, plus 5 B3 rejections, 5 B4 wrong credits,
and 5 B5 penalties. The initial 40 use seed-490 stratified selection; the 15 targeted follow-ups are
fixed by ID. IDs and review judgments are committed in `method_b_audit.json`. Full texts and scores
are joined locally in `reviewed_examples.jsonl`. This is a **single-agent, targeted
qualitative audit**, not independent human review or a population precision/recall
estimate. The first 40 include 20 credited and 20 uncredited responses; the other
15 stress specific failure modes. The audit uses the full retained population to
include suspected export failures, not just the cleaner metric subset.

Among the 20 credited cases, the matched numerical computations are verified, but
only 15 were judged clearly useful partial steps; 2 had ambiguous prose/context
and 3 B2-only cases showed semantic role/unit mismatches. These judgments are about
visible evidence, not proof that the model reasoned correctly. Among the 20
uncredited cases, 16 appropriately lack a supported completed calculation, 3 miss
useful reasoning due to reference form/coverage, and 1 has an annotation gap plus
entity-label errors. No new production arithmetic precision bug was established.

| Audit ID (`step:problem:sample`) | Observation | Implication |
|---|---|---|
| `213:3415:3` | `4*9=36`, `3*5=15`, `5*7=35` correctly compute candy costs; reference writes operands reversed. Final omits the pinata. B1=0, B2=0.15. | Genuine B2 partial credit, not full-answer correctness. |
| `348:5578:13` | Correct elephant subtotals reach 280, then an extra addition gives 350. B1=0.10, B2=0.20. | Useful prefix supervision even when a later step is wrong. |
| `202:3247:0` | `30*4=120` matches the reference's **student** count, but the response calls it **buns**. | Numeric/tree matching alone does not verify semantic grounding. B3 also accepts it. |
| `196:3143:11` | Reused 2 and 3 numerically match chocolate computations with wrong units/store roles; final 6 happens to match. | Correct final labels do not establish valid reasoning; B2/B3 can credit numeric aliases. |
| `211:3391:11` | 100*4 season cost is described as monthly savings. | Another B2-only semantic alias; B3 cannot reject equal values with wrong meanings. |
| `223:3575:1` | 500*.8=400 matches the insurance **discount**, but prose calls 400 the lenses' cost. | B1 is also vulnerable to context mismatch; operand order is not semantic protection. |
| `182:2927:7` | Valid weekly base-pay regrouping and `18*1.5=27` receive zero because reference decomposes them differently; later overtime is wrong. | Conservative false negative; do not silently add reassociation or arbitrary constant folding. |
| `193:3103:6` | Valid weekly salary/tax/net calculations precede an omitted annual factor; reference annotates annual gross in one expression. | Useful prefix missed because granularity differs, not incorrect calculator arithmetic. |
| `391:6266:1` | Correct solution uses implicit multiplication and reference has no annotations. | Both annotation coverage and notation limit recall. |
| `188:3015:6`, `179:2871:5` | Correct steps use 'thrice' or 'one-fourth', absent from B3's limited roots. | Quantity provenance creates false negatives. |
| `330:5282:11` | Entire correct discount solution uses .20/.30 converted from percentages. | B3 rejects valid inferred quantities; extending a lexicon is not a semantic guarantee. |
| `377:6042:6`, `323:5178:3`, `93:1495:10` | Last-result/final consistency reports 80 cookies as calories, 50 total customers as customers/car, or intermediate 11 as total 33. | B4 can reinforce the Part 3 failure it is intended to fix. |
| `49:788:0`, `227:3647:11` | Ordinary tool floats `110.00000000000001` / `50.333333333333336` trigger repeated-digit penalties. | B5 confuses deterministic floating output with degenerate generation. |
| `340:5450:10` | Last calculator call is incomplete after a long response. | Could be model termination or the 256-token cap; metadata does not decide. |
| `70:1122:5` | Malformed output marker plus text from an unrelated raking problem. | Possible export corruption, not evidence of a model failure. |

The five targeted B3 rejections all remove useful reference-aligned calculations.
The three B2-only semantic mismatches above survive B3. On all retained responses,
naive unused-numeric-value flags affect **1,500/4,010 correct responses (37.41%)**;
ungrounded-operand flags affect **742/4,010 (18.50%)**. Numeric sets cannot distinguish
multiple entities sharing a value, distractors, unit conversions or implicit steps.
Consequently omitted/invented-quantity penalties are not justified by these detectors.
This rejects the evaluated numeric-provenance proxy, not every possible future
semantic grounding method. A better lexicon alone would not resolve entity/unit aliases.

Ungated valid-tool/final consistency occurs in **10,054** responses, including
**6,199 incorrect** ones. It is counted only diagnostically, never rewarded. Even
reference-gated B4 credits **507 incorrect** responses in the retained subset.
Its high correctness correlation largely reflects near-answer overlap, while its
primary all-wrong differentiation (74/353) is far below B1/B2.

B5 flags 347 syntactically incomplete calls and 491 repeated-digit responses (with
overlap), including penalties on 16 originally correct responses. Those are detector
hits, not verified degenerate/truncated generations. The exports have no token
counts, finish reasons, termination token IDs or sampling-cap metadata. **All
termination-versus-cap causal classifications are unknown.** Character length or a
sentence ending is not a reliable substitute. Ordinary non-tool/prose truncation
is not penalized by this deliberately limited B5.

Reference-substep rewards can also be gamed by emitting a list of valid known
subcomputations without solving the question. Repeating the same step cannot inflate
B1-B3, but distinct matched irrelevant-context steps remain possible. Synthetic tests
cover unsafe code, arbitrary true equations, invalid outputs, duplicate steps,
subtraction/division swaps, reassociation and numerical/unit alias risks. No automated
suite can establish semantic precision for natural-language quantities by itself.

## Proposed change for review; not applied to production

1. Retain B1's restricted parser, exact-rational computation, result tolerance,
   weight cap, and annotated-reference requirement.
2. Add a small tree canonicalizer that recursively sorts **binary Add/Mult children
   only**, applied identically to validated reference and generated steps.
3. Deduplicate after canonicalization and retain a score equal to matched unique
   steps divided by eligible unique reference steps, times the existing weight.
4. Keep subtraction/division order, association and every other AST operation
   unchanged. Include tests for repeated/equivalent reference annotations and the
   resulting denominator change. Do not infer entities/units from coincident numbers.
5. Give any approved B2 experiment a fresh explicit version/name and compare it with
   B1 using the same SFT checkpoint and sampling/optimizer setup; do not relabel B1
   results as B2. Leave Method A and baseline correctness unchanged.

Before any GPU training, the next experiment is **local, blinded semantic review**
of a larger question-stratified set of B2-only/newly credited steps, B1-only/fraction
changes, and matched negative controls. Use independent human annotation of relevant
quantity/entity/unit use, rather than the Part 3 heuristic labels or final correctness
alone. Report response- and question-level agreement/precision with uncertainty;
predeclare an acceptable error margin and withhold adoption if B2 introduces an
unacceptable semantic loss. A proposed margin is a review decision, not something
chosen post hoc to make this sample pass. Also validate repaired rollouts and capture
future finish-reason/token-count metadata before reconsidering B5. No unrelated
B3-B5 bonuses should be bundled into Method B on current evidence.

If review approves B2, compare baseline, A, B2 and A+B2 in separately authorized
controlled runs, retaining B1 as the direct ablation. This document does not authorize
any cloud run or claim learning/test-accuracy improvements.

## Reproduction and outputs: local CPU only

Python 3.12 + pytest; NumPy is needed to reproduce the seed-42 index cross-check.
No torch, datasets, pyarrow, model weights or cloud authentication is needed.
The recovery CLI requires a 7,473-row train file, refusing a test-sized input. Its
optional network path is restricted to the pinned public **train.jsonl** URL.

```powershell
# Local recovery; add --download-train only if the local training file is absent.
py -3.12 a4/recover_gsm8k_references.py C:/Users/wendy/Documents/a4-rollout-analysis/a4-rl3.jsonl C:/Users/wendy/Documents/a4-rollout-analysis/a4-rl3-resume.jsonl --train-file a4/.reward-analysis/method-b-data/gsm8k-main-train.jsonl --output-dir a4/.reward-analysis/my-reference-recovery

# Local comparison; use a fresh output directory to avoid overwriting artifacts.
py -3.12 a4/compare_method_b.py a4/.reward-analysis/my-reference-recovery/recovered.jsonl --output-dir a4/.reward-analysis/my-method-b-comparison --audit-notes a4/method_b_audit.json

py -3.12 -m pytest a4/test_rewards.py a4/test_arithmetic_reward.py a4/test_analyze_reward_signals.py a4/test_reward_integration.py a4/test_modal_launch.py a4/test_method_b_research.py -q
py -3.12 -m compileall -q a4
```

Completed artifacts: `a4/.reward-analysis/method-b-recovery-v1/` and
`a4/.reward-analysis/method-b-selection-final/`, both ignored by Git. Outputs include
`recovered.jsonl`, `recovery.json`, `summary.json`, `summary.csv`, `breakdowns.csv`,
`scored_responses.jsonl`, fixed-seed `audit_examples.jsonl` and joined
`reviewed_examples.jsonl`. Raw text is kept local; committed audit notes contain IDs
and interpretation only. Existing output directories and dataset files are never
replaced. `analyze_reward_signals.load_records` only gains the question field in its
returned local-analysis records; production reward modules are unchanged.

The relevant local suite passed **161 tests**, and `compileall -q a4` passed. The
separate existing FoX cache tests require missing local torch and were not run here;
no dependencies/models were downloaded to satisfy them.

**Zero Modal GPU jobs and zero billable cloud compute operations were executed.**
No Modal commands, remote/spawn calls, workspace authentication changes, Laith
workspace access, model loads, paid APIs or GSM8K test-label tuning occurred.
