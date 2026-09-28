# Preference consistency of local LLM judges

This repository is a small, local study of a practical question in modern ML systems:

> When we ask a model (or a human) “which answer is better?”, how stable is that judgment if we change **order**, **wording**, or **extra instructions**—including instructions that push the judge to agree with a stated user preference?

It was developed by someone learning these topics more deeply and packaging the experiments so others can reproduce them on a laptop with **open-weight** models only (no paid APIs).

We re-implement **evaluation protocols** from published work on a seeded slice of [Anthropic HH-RLHF](https://arxiv.org/abs/2204.05862), using a prompted local judge (default: `llama3.2:3b` via Ollama). This is **not** a trained reward model and **not** a claim of state-of-the-art numbers matching GPT-4 papers.

## Who this is for

- People comfortable with Python and basic ML who are newer to AI safety / alignment evaluation
- Readers who want a runnable walkthrough with paper links, not only a black-box script

## What you will learn

- What preference pairs (`chosen` / `rejected`) are and how they enter RLHF-style pipelines ([Bai et al. 2022](https://arxiv.org/abs/2204.05862))
- LLM-as-a-judge vs a trained reward model ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685))
- Measuring **position bias** by swapping A/B ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685), [Wang et al. 2023](https://arxiv.org/abs/2305.17926))
- Lightweight **sycophancy / agreeableness** probes on the judge ([Perez et al. 2022](https://arxiv.org/abs/2212.09251))
- Agreement, flip rate, Cohen’s κ, Wilson confidence intervals on small *n*

## Start here

1. **Install & data:** follow **[docs/SETUP.md](docs/SETUP.md)** (`make setup`, `make bootstrap-judge`, `make fetch-data`).
2. **Learn by doing:** open **[notebooks/walkthrough_local_judge.ipynb](notebooks/walkthrough_local_judge.ipynb)** (concept-first; includes “try your own prompt” cells).
3. **Batch eval:** `make local-smoke` → read [`results/local/results.md`](results/local/results.md).

## Papers we approximate

| Paper | Idea we reuse | What we do **not** reproduce |
|---|---|---|
| [Bai et al. 2022 — HH-RLHF](https://arxiv.org/abs/2204.05862) | Public helpful/harmless preference pairs as labels | Anthropic’s trained preference model / full RLHF |
| [Zheng et al. 2023 — LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) | Pairwise judge prompts; position effects; dual-order thinking | GPT-4 judge, full MT-Bench, Chatbot Arena |
| [Wang et al. 2023 — LLMs are not Fair Evaluators](https://arxiv.org/abs/2305.17926) | Order/position bias in pairwise LLM judges | Their API models and full benchmark |
| [Perez et al. 2022 — Discovering language model behaviors…](https://arxiv.org/abs/2212.09251) | Sycophancy: models may agree with a user’s stated view | Full model-written evaluation suite; we only bias the *judge* prompt |

**Paraphrase + conflicting-instruction** conditions extend the same harness (rule-based paraphrases; short vs thorough instructions).

Honest gap: a 3B prompted judge ≠ a large trained reward model ≠ GPT-4-as-judge. Compare **direction** of effects, not absolute percentages in those papers.

## Sample findings

After a real local run, see [`results/local/results.md`](results/local/results.md) for tables (accuracy, position flip rate, sycophancy agreement vs baseline) with Wilson CIs. Numbers will vary by model and seed; the write-up is a snapshot, not a leaderboard.

## Reproduce (after SETUP)

```bash
make doctor
make fetch-data
make local-smoke    # HH n=50: baseline, swap, sycophancy → results/local/
make local-eval     # all conditions on HH n=50
make test           # offline unit tests
make smoke          # fixture + dry-run (no network, no Ollama)
```

Config: [`configs/default.yaml`](configs/default.yaml). Prompts: [`configs/prompts/`](configs/prompts/).

## Conditions

- `baseline` — judge A vs B  
- `position_swap` — A↔B; metrics use **content identity** (`chosen`/`rejected`), not the letter  
- `paraphrase_*` — rule-based formatting/lexical changes  
- `conflict_short` / `conflict_thorough` — extra length preferences  
- `sycophancy_prefer_a` / `prefer_b` / `agree_user` — Perez-inspired judge bias  

## Layout

```
docs/SETUP.md                         ← install, Ollama, HH cache
notebooks/walkthrough_local_judge.ipynb  ← teaching + playground
configs/default.yaml
configs/prompts/
src/preference_consistency/
data/fixtures/hh_tiny.jsonl           ← offline fallback
data/cache/                           ← HH subset (gitignored)
results/local/                        ← example write-ups
```

## Limitations

- Small *n*; wide confidence intervals  
- Prompted small model, not a reward head  
- Sycophancy probes are judge-prompt biases, not Perez’s full pipeline  
- Rule-based paraphrases understate rich paraphrase sensitivity  

## License / citation

Code in this repository is released under the [MIT License](LICENSE).

If you reuse this for teaching or further experiments, please cite the original papers above and note that this repo is a small-scale local approximation of their *evaluation ideas*. The HH-RLHF data itself remains subject to Anthropic’s dataset terms on Hugging Face.
