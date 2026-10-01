# Published study: Qwen2.5 7B, n=100

`make study` on HH-RLHF `helpful-base`, seed 42, temperature 0. Narrative and comparison to the matched 3B run: [`docs/FINDINGS.md`](../../docs/FINDINGS.md).

- `results.md` — tables
- `figures/` — Wilson-interval charts, including the 3B vs 7B comparison
- `demos.json` — one pair with the prompt and raw completion under several conditions
- `judgments.jsonl` — full rows, gitignored; rerun `make study` to regenerate
