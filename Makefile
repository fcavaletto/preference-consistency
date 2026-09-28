PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup smoke run test doctor bootstrap-judge fetch-data local-smoke local-eval

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

# Real HH (n=50): baseline + position swap + sycophancy
local-smoke:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--n-samples 50 \
		--conditions baseline position_swap sycophancy_prefer_a sycophancy_prefer_b sycophancy_agree_user \
		--output-dir results/local

# Full MVP conditions on HH n=50
local-eval:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--n-samples 50 \
		--output-dir results/local

run:
	$(PYTHON) -m preference_consistency.run_eval \
		--config configs/default.yaml \
		--n-samples 200 \
		--output-dir results

test:
	$(PYTHON) -m pytest -q
