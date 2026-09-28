# Preference consistency results

Small-scale **prompted local judge** study. Not a trained reward model and not SOTA.

## Setup

- Model: `dry-run`
- Dry-run: `True`
- Temperature: `0.0` (greedy; documented even if a paper used sampling)
- n pairs: `8`  seed: `42`  data: `fixture`
- Parse errors: `0` / `56` judgments

## Metrics

| Condition | Accuracy vs HH label | P(pick first slot) |
|---|---|---|
| `baseline` | 1.000 (8/8) 95% CI [0.676, 1.000] | 0.500 (4/8) 95% CI [0.215, 0.785] |
| `conflict_short` | 0.000 (0/8) 95% CI [0.000, 0.324] | 0.500 (4/8) 95% CI [0.215, 0.785] |
| `conflict_thorough` | 1.000 (8/8) 95% CI [0.676, 1.000] | 0.500 (4/8) 95% CI [0.215, 0.785] |
| `paraphrase_lexical` | 1.000 (8/8) 95% CI [0.676, 1.000] | 0.500 (4/8) 95% CI [0.215, 0.785] |
| `paraphrase_role_markup` | 1.000 (8/8) 95% CI [0.676, 1.000] | 0.500 (4/8) 95% CI [0.215, 0.785] |
| `paraphrase_whitespace` | 1.000 (8/8) 95% CI [0.676, 1.000] | 0.500 (4/8) 95% CI [0.215, 0.785] |
| `position_swap` | 0.500 (4/8) 95% CI [0.215, 0.785] | 1.000 (8/8) 95% CI [0.676, 1.000] |

### Position swap (Zheng / Wang-style)

- Content-level agreement after A↔B: 0.500 (4/8) 95% CI [0.215, 0.785]
- **Flip rate**: 0.500 (4/8) 95% CI [0.215, 0.785]
- Cohen's κ (baseline vs swap, content id): `0.000`

### Paraphrase self-consistency

- `paraphrase_whitespace` vs baseline: 1.000 (8/8) 95% CI [0.676, 1.000]; κ = `1.000`
- `paraphrase_role_markup` vs baseline: 1.000 (8/8) 95% CI [0.676, 1.000]; κ = `1.000`
- `paraphrase_lexical` vs baseline: 1.000 (8/8) 95% CI [0.676, 1.000]; κ = `1.000`

### Conflicting instructions

- `conflict_short` vs `conflict_thorough` content agreement: 0.000 (0/8) 95% CI [0.000, 0.324]; κ = `0.000`

## Comparison to papers (order of magnitude, not a copy)

| Source | Reported pattern | This run |
|---|---|---|
| Bai et al. 2022 (HH-RLHF) | Human `chosen`/`rejected` pairs as labels | Accuracy vs those labels on a tiny slice |
| Zheng et al. 2023 (LLM-as-judge / MT-Bench) | Non-zero position bias; they recommend judging both orders | Our flip rate / P(first) above; GPT-4 in their paper is far more stable than a 3B local judge |
| Wang et al. 2023 (not fair evaluators) | Pairwise LLM judges systematically prefer a position | Same swap protocol; we expect **higher** flips than their GPT-family tables |

### Why numbers will differ from the papers

- **Prompted 3B–8B judge ≠ trained Bradley–Terry RM** on 52B+ (Anthropic) or GPT-4 judges.
- **n is 8–500**, not MT-Bench/Chatbot Arena scale; CIs will be wide.
- **HH helpfulness transcripts** are not MT-Bench multi-turn scoring items.
- **Temperature 0** local decoding vs API sampling used in some judge papers.
- **Rule-based paraphrases** are conservative vs LLM paraphrases of the query.
- **This file was produced with `--dry-run`**. Heuristic length/position rules, not a model.

## Limitations

See the README. Do not cite these figures as a reproduction of GPT-4 or Anthropic RM results.
