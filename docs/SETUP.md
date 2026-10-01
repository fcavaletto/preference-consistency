# Setup

Install the Python package, a local Ollama judge, and (by default) a cached slice of Anthropic’s HH-RLHF preference data. The teaching notebook assumes this is done; it does not walk through Homebrew or disk space.

## Prerequisites

- macOS with Apple Silicon recommended (works anywhere Ollama runs)
- **Python 3.11+** (Homebrew: `brew install python@3.12`)
- Disk: ~5 GB for `qwen2.5:7b`, plus tens of MB for the HH train gzip on first fetch
- Memory: a machine with about 16 GB RAM is comfortable for a 7B Q4 judge (the 3B model still works for the scale comparison)
- Network for first-time model pull and HH download

## 1. Clone and create the venv

```bash
git clone <your-fork-or-repo-url> preference-consistency
cd preference-consistency
make setup
```

This creates `.venv` and installs the package in editable mode (`httpx`, `pyyaml`, `pandas`, pytest, ipykernel).

## 2. Local judge (Ollama)

```bash
make bootstrap-judge   # brew install ollama if needed; start serve; pull qwen2.5:7b
make doctor            # must print: Default judge ready: qwen2.5:7b
```

Keep the daemon running: open the **Ollama app**, or leave `ollama serve` in a terminal. Homebrew’s CLI install does not always keep a background service. From this repo you can also run `make doctor` / `make ollama`, which start `ollama serve` if port 11434 is down. The teaching notebook does the same on its setup cell.

Default model tag is set in [`configs/default.yaml`](../configs/default.yaml) (`qwen2.5:7b`, temperature `0`, fixed seed). The matched 3B comparison uses `llama3.2:3b`, which you can pull with `ollama pull llama3.2:3b` if it is not already local.

## 3. Preference data (HH-RLHF by default)

```bash
make fetch-data
```

This downloads Anthropic **HH-RLHF** `helpful-base` train ([Bai et al. 2022](https://arxiv.org/abs/2204.05862); dataset on Hugging Face as `Anthropic/hh-rlhf`), reservoir-samples the config default (**200** pairs, seed 42), randomizes A/B slots, and caches:

`data/cache/hh_subset_helpful-base_seed42_n200.jsonl`

`make study` uses its own cache of **100** pairs (`hh_subset_helpful-base_seed42_n100.jsonl`). Later runs reuse the matching cache. To rebuild, delete that file and fetch again.

**Offline / CI without HH:** use the bundled synthetic fixture:

```bash
make smoke   # --fixture --dry-run, 8 pairs, no Ollama
```

## 4. Notebook kernel

In Cursor / Jupyter, open [`notebooks/walkthrough_local_judge.ipynb`](../notebooks/walkthrough_local_judge.ipynb) and select:

- **Python (.venv preference-consistency)**, or  
- the interpreter at `.venv/bin/python`

If the kernel is missing after `make setup`:

```bash
.venv/bin/python -m ipykernel install --user \
  --name preference-consistency \
  --display-name "Python (.venv preference-consistency)"
```

## 5. First real eval

```bash
make doctor
make fetch-data
make local-smoke
```

`local-smoke` runs n=50 HH pairs with baseline, position swap, sycophancy, and verbosity → `results/local/`.

Published study (this is the run the walkthrough charts use):

```bash
make study       # qwen2.5:7b, n=100, all conditions → results/qwen2.5-7b/
make study-3b    # same pairs and conditions, llama3.2:3b → results/llama3.2-3b-matched/
make figures     # Wilson-interval charts, including the 3B vs 7B comparison
```

Shorter full condition set at n=50:

```bash
make local-eval
```

## Make targets (cheat sheet)

| Target | What it does |
|---|---|
| `make setup` | venv + install |
| `make bootstrap-judge` | Ollama + pull `qwen2.5:7b` |
| `make doctor` | check Ollama + model |
| `make fetch-data` | HH subset cache only |
| `make smoke` | fixture + dry-run (offline) |
| `make local-smoke` | HH n=50, baseline/swap/sycophancy/verbosity |
| `make local-eval` | HH n=50, all conditions |
| `make study` | published 7B run, HH n=100 |
| `make study-3b` | matched 3B run, same 100 pairs |
| `make figures` | charts under `results/qwen2.5-7b/figures/` |
| `make notebook` | regenerate the teaching notebook |
| `make run` | HH n=200, all conditions |
| `make test` | pytest (no network) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama` | Start the app or `ollama serve`; then `make doctor` |
| Model missing | `ollama pull qwen2.5:7b` |
| HH download fails | Check network / Hugging Face access; retry `make fetch-data` |
| Many `parse_error` | Inspect raw text in `judgments.jsonl`; try `json_schema: false` in config |
| Want offline only | `--fixture` on the CLI, or `make smoke` |

## What not to expect from setup

- No paid APIs  
- No reward-model training  
- Default judge is a prompted 7B model, not a flagship API model  

When setup succeeds, open the teaching notebook and stay there for concepts and experiments.
