"""Tests for condition collection and prompts (no network)."""

from __future__ import annotations

from pathlib import Path

import yaml

from preference_consistency.prompts import load_text
from preference_consistency.run_eval import collect_conditions, repo_root


def test_collect_conditions_includes_sycophancy() -> None:
    root = repo_root()
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    names = collect_conditions(cfg)
    assert "baseline" in names
    assert "position_swap" in names
    assert "sycophancy_prefer_a" in names
    assert "sycophancy_prefer_b" in names
    assert "sycophancy_agree_user" in names
    assert "conflict_short" in names


def test_sycophancy_prompt_files_exist() -> None:
    root = repo_root()
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    for rel in (cfg["conditions"]["sycophancy"] or {}).values():
        text = load_text(root / rel)
        assert len(text) > 20
