# Setup

Install the Python package, a local Ollama judge, and (by default) a cached slice of Anthropic’s HH-RLHF preference data. The teaching notebook assumes this is done; it does not walk through Homebrew or disk space.

## Prerequisites

- macOS with Apple Silicon recommended (works anywhere Ollama runs)
- **Python 3.11+** (Homebrew: `brew install python@3.12`)
- Disk: ~2 GB for `llama3.2:3b`, plus tens of MB for the HH train gzip on first fetch
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
make bootstrap-judge   # brew install ollama if needed; start serve; pull llama3.2:3b
make doctor            # must print: Default judge ready: llama3.2:3b
```

Keep the daemon running: open the **Ollama app**, or leave `ollama serve` in a terminal. Homebrew’s CLI install does not always keep a background service.

Default model tag is set in [`configs/default.yaml`](../configs/default.yaml) (`llama3.2:3b`, temperature `0`, fixed seed). Pulling larger models is optional and not required for the walkthrough.

## 3. Preference data (HH-RLHF by default)

```bash
make fetch-data
```

This downloads Anthropic **HH-RLHF** `helpful-base` train ([Bai et al. 2022](https://arxiv.org/abs/2204.05862); dataset on Hugging Face as `Anthropic/hh-rlhf`), reservoir-samples **50** pairs (seed 42), randomizes A/B slots, and caches:

`data/cache/hh_subset_helpful-base_seed42_n50.jsonl`

Later runs reuse the cache (no re-download). To rebuild, delete that file and run `make fetch-data` again.

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

`local-smoke` runs n=50 HH pairs with **baseline**, **position swap**, and **sycophancy** conditions → `results/local/`.

Full condition set (also paraphrases + conflicting instructions):

```bash
make local-eval
```

## Make targets (cheat sheet)

| Target | What it does |
|---|---|
| `make setup` | venv + install |
| `make bootstrap-judge` | Ollama + pull 3B |
| `make doctor` | check Ollama + model |
| `make fetch-data` | HH subset cache only |
| `make smoke` | fixture + dry-run (offline) |
| `make local-smoke` | HH n=50, baseline/swap/sycophancy |
| `make local-eval` | HH n=50, all conditions |
| `make run` | HH n=200, all conditions |
| `make test` | pytest (no network) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama` | Start the app or `ollama serve`; then `make doctor` |
| Model missing | `ollama pull llama3.2:3b` |
| HH download fails | Check network / Hugging Face access; retry `make fetch-data` |
| Many `parse_error` | Inspect raw text in `judgments.jsonl`; try `json_schema: false` in config |
| Want offline only | `--fixture` on the CLI, or `make smoke` |

## What not to expect from setup

- No paid APIs  
- No reward-model training  
- Default judge is a small prompted model, not GPT-4  

When setup succeeds, open the teaching notebook and stay there for concepts and experiments.
