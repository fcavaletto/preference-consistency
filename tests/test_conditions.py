"""Tests for condition collection and prompts (no network)."""

from __future__ import annotations

from pathlib import Path

import yaml

from preference_consistency.data import load_fixture
from preference_consistency.metrics import dual_order_consistency, slot_attraction
from preference_consistency.perturbations import apply_verbosity, bloat_text
from preference_consistency.prompts import load_text
from preference_consistency.run_eval import collect_conditions, prepare_condition, repo_root


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


def test_collect_conditions_includes_verbosity_and_user_sycophancy() -> None:
    root = repo_root()
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    names = collect_conditions(cfg)
    assert "verbosity_bloat_a" in names
    assert "verbosity_bloat_b" in names
    assert "sycophancy_user_prefers_a" in names
    assert names.index("sycophancy_user_prefers_a") > names.index("sycophancy_prefer_a")


def test_bloat_repeats_text_without_changing_the_other_slot() -> None:
    root = repo_root()
    pair = load_fixture(root / "data/fixtures/hh_tiny.jsonl", n_samples=1, seed=0)[0]
    bloated = bloat_text(pair.response_a)
    assert pair.response_a.strip() in bloated
    assert "without adding new information" in bloated
    assert len(bloated) > len(pair.response_a)
    only_a = apply_verbosity(pair, "bloat_a")
    assert only_a.response_b == pair.response_b
    assert only_a.label == pair.label
    assert only_a.chosen == pair.chosen
    only_b = apply_verbosity(pair, "bloat_b")
    assert only_b.response_a == pair.response_a
    assert pair.response_b.strip() in only_b.response_b


def test_user_sycophancy_is_user_text_not_system() -> None:
    root = repo_root()
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    pair = load_fixture(root / "data/fixtures/hh_tiny.jsonl", n_samples=1, seed=1)[0]
    cond = cfg["conditions"]
    work, extra_system, extra_user = prepare_condition(
        pair,
        "sycophancy_user_prefers_a",
        root=root,
        conflict_files=cond["conflicting_instructions"],
        sycophancy_files=cond["sycophancy"],
        sycophancy_user_files=cond["sycophancy_user"],
    )
    assert work.response_a == pair.response_a
    assert extra_system == ""
    assert "Response A" in extra_user
    assert "agree" in extra_user.lower()


def test_dual_order_accuracy_uses_only_agreeing_pairs() -> None:
    stats = dual_order_consistency(
        ["chosen", "rejected", "chosen", "chosen"],
        ["chosen", "rejected", "rejected", "rejected"],
        [True, False, True, False],
    )
    assert stats["n_consistent"] == 2
    assert stats["accuracy_consistent"].k == 1
    assert stats["accuracy_consistent"].n == 2
    assert stats["flip"].k == 2
    assert stats["agreement"].k == 2


def test_slot_attraction_ignores_cases_already_on_the_slot() -> None:
    attracted = slot_attraction(["A", "A", "B", "B"], ["A", "B", "B", "A"], "B")
    assert attracted.k == 1
    assert attracted.n == 2


def test_sycophancy_prompt_files_exist() -> None:
    root = repo_root()
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    for rel in (cfg["conditions"]["sycophancy"] or {}).values():
        text = load_text(root / rel)
        assert len(text) > 20
