# Preference consistency results

Small-scale **prompted local judge** study. Not a trained reward model and not SOTA.

## Setup

- Model: `llama3.2:3b`
- Dry-run: `False`
- Temperature: `0.0` (greedy; documented even if a paper used sampling)
- n pairs: `50`  seed: `42`  data: `download`
- Parse errors: `0` / `250` judgments

## Metrics

| Condition | Accuracy vs HH label | P(pick first slot) |
|---|---|---|
| `baseline` | 0.520 (26/50) 95% CI [0.385, 0.652] | 0.020 (1/50) 95% CI [0.004, 0.105] |
| `position_swap` | 0.600 (30/50) 95% CI [0.462, 0.724] | 0.100 (5/50) 95% CI [0.043, 0.214] |
| `sycophancy_agree_user` | 0.500 (25/50) 95% CI [0.366, 0.634] | 0.040 (2/50) 95% CI [0.011, 0.135] |
| `sycophancy_prefer_a` | 0.540 (27/50) 95% CI [0.404, 0.670] | 0.080 (4/50) 95% CI [0.032, 0.188] |
| `sycophancy_prefer_b` | 0.500 (25/50) 95% CI [0.366, 0.634] | 0.000 (0/50) 95% CI [0.000, 0.071] |

### Position swap (Zheng / Wang-style)

- Content-level agreement after A↔B: 0.120 (6/50) 95% CI [0.056, 0.238]
- **Flip rate**: 0.880 (44/50) 95% CI [0.762, 0.944]
- Cohen's κ (baseline vs swap, content id): `-0.774`

### Sycophancy / judge bias (Perez-style, lightweight)

We bias the *judge* with an extra instruction (user prefers A/B, or “agree with the user”). This is **not** a full reproduction of Perez et al.'s model-written evaluation suite ([arXiv:2212.09251](https://arxiv.org/abs/2212.09251)); it is a small protocol to ask: does the judge follow the stated preference over content?

- `sycophancy_agree_user` vs baseline: agreement 0.980 (49/50) 95% CI [0.895, 0.996]; flip 0.020 (1/50) 95% CI [0.004, 0.105]; κ = `0.960`
- `sycophancy_prefer_a` vs baseline: agreement 0.940 (47/50) 95% CI [0.838, 0.979]; flip 0.060 (3/50) 95% CI [0.021, 0.162]; κ = `0.880`
  - P(verdict=A | prefer_a): 0.080 (4/50) 95% CI [0.032, 0.188]
- `sycophancy_prefer_b` vs baseline: agreement 0.980 (49/50) 95% CI [0.895, 0.996]; flip 0.020 (1/50) 95% CI [0.004, 0.105]; κ = `0.960`
  - P(verdict=B | prefer_b): 1.000 (50/50) 95% CI [0.929, 1.000]

## Comparison to papers (order of magnitude, not a copy)

| Source | Reported pattern | This run |
|---|---|---|
| [Bai et al. 2022 (HH-RLHF)](https://arxiv.org/abs/2204.05862) | Human `chosen`/`rejected` pairs | Accuracy vs those labels on a seeded subset |
| [Zheng et al. 2023 (LLM-as-judge)](https://arxiv.org/abs/2306.05685) | Position bias; judge both orders | Flip rate / P(first) above |
| [Wang et al. 2023 (not fair evaluators)](https://arxiv.org/abs/2305.17926) | Pairwise judges prefer a position | Same swap protocol; expect **higher** flips on 3B local judges |
| [Perez et al. 2022 (sycophancy)](https://arxiv.org/abs/2212.09251) | Models change answers to agree with users | Judge-side bias prompts; see sycophancy section |

### Why numbers will differ from the papers

- **Prompted 3B chat model ≠ trained Bradley–Terry RM** or GPT-4-as-judge.
- **n is tens–hundreds**, not MT-Bench / Arena scale; CIs will be wide.
- **HH helpfulness pairs** are not MT-Bench multi-turn scoring items.
- **Temperature 0** local decoding vs API sampling in some judge papers.
- **Rule-based paraphrases** understate LLM paraphrase sensitivity.
- **Sycophancy conditions** bias the judge prompt; they are not Perez's full dataset pipeline.

## Limitations

See the README. Do not cite these figures as a copy of GPT-4 or Anthropic RM results.
