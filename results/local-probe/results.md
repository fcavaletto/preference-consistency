# Preference consistency results

Small-scale **prompted local judge** study. Not a trained reward model and not SOTA.

## Setup

- Model: `llama3.2:3b`
- Dry-run: `False`
- Temperature: `0.0` (greedy; documented even if a paper used sampling)
- n pairs: `1`  seed: `42`  data: `fixture`
- Parse errors: `0` / `1` judgments

## Metrics

| Condition | Accuracy vs HH label | P(pick first slot) |
|---|---|---|
| `baseline` | 0.000 (0/1) 95% CI [0.000, 0.793] | 0.000 (0/1) 95% CI [0.000, 0.793] |

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

## Limitations

See the README. Do not cite these figures as a reproduction of GPT-4 or Anthropic RM results.
