# Notebooks

1. **Setup first:** [docs/SETUP.md](../docs/SETUP.md) (venv, Ollama, Qwen2.5 7B, HH cache).
2. **Teaching notebook:** [walkthrough_local_judge.ipynb](walkthrough_local_judge.ipynb). Read it top to bottom. Charts use the saved `make study` run. Cells you are meant to run yourself call the model or open a widget; the published copy leaves those unexecuted.
3. **After an eval:** [explore_results.ipynb](explore_results.ipynb) loads `results/qwen2.5-7b/judgments.jsonl`.

Regenerate the teaching notebook from source with:

```bash
make notebook
```
