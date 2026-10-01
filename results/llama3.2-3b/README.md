# Archived Llama 3.2 3B snapshot (n=50)

This is the earlier `make local-smoke` run, kept because it is a clean teaching example of a confound: the judge picked slot A on 1 of 50 pairs and flipped content on 44 of 50 swaps. A “prefer B” probe on that model is not evidence of sycophancy until you subtract that baseline.

The fair scale comparison is `make study-3b`, which repeats the full 7B protocol on the same 100 pairs and writes `results/llama3.2-3b-matched/`.

Raw `judgments.jsonl` is gitignored. `results.md` is the committed table.
