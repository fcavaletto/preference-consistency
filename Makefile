PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup smoke run test doctor bootstrap-judge fetch-data local-smoke local-eval study study-3b figures notebook

setup:
	@if command -v python3.12 >/dev/null 2>&1; then PY=python3.12; \
	elif command -v python3.11 >/dev/null 2>&1; then PY=python3.11; \
	else echo "Need Python 3.11+. On macOS: brew install python@3.12"; exit 1; fi; \
	$$PY -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

# Offline / CI: synthetic fixture, no Ollama, no HH download
smoke:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--fixture \
		--dry-run \
		--n-samples 8 \
		--output-dir results

doctor:
	$(PYTHON) -m preference_consistency.check --config configs/default.yaml

bootstrap-judge:
	chmod +x scripts/bootstrap_local_judge.sh
	./scripts/bootstrap_local_judge.sh
	$(PYTHON) -m preference_consistency.check

# Download/cache default HH subset (no model calls)
fetch-data:
	$(PYTHON) -m preference_consistency.fetch_data --config configs/default.yaml

# Real HH (n=50): baseline, swap, sycophancy, verbosity
local-smoke:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--n-samples 50 \
		--conditions baseline position_swap sycophancy_prefer_a sycophancy_prefer_b sycophancy_agree_user sycophancy_user_prefers_a verbosity_bloat_a verbosity_bloat_b \
		--output-dir results/local

# Full condition set on HH n=50
local-eval:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--n-samples 50 \
		--output-dir results/local

# Published study: Qwen2.5 7B, n=100, all conditions
study:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--n-samples 100 \
		--output-dir results/qwen2.5-7b

# Same 100 pairs and conditions on the previous 3B judge
study-3b:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--model llama3.2:3b \
		--n-samples 100 \
		--output-dir results/llama3.2-3b-matched

figures:
	$(PYTHON) -m preference_consistency.plots \
		--judgments results/qwen2.5-7b/judgments.jsonl \
		--out results/qwen2.5-7b/figures \
		--title "qwen2.5:7b" \
		--compare results/llama3.2-3b-matched/judgments.jsonl \
		--compare-label "llama3.2:3b"

notebook:
	$(PYTHON) scripts/build_walkthrough_nb.py

run:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--n-samples 200 \
		--output-dir results

test:
	$(PYTHON) -m pytest -q
