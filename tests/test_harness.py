"""Unit tests that do not require Ollama."""

from __future__ import annotations

from pathlib import Path

from preference_consistency.data import load_fixture, split_prompt_and_response
from preference_consistency.infer import DryRunClient, parse_verdict
from preference_consistency.metrics import cohens_kappa, wilson_interval
from preference_consistency.perturbations import position_swap


FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "hh_tiny.jsonl"


def test_split_hh_transcript() -> None:
    prompt, resp = split_prompt_and_response(
        "Human: Hi?\n\nAssistant: Hello.\n\nHuman: More?\n\nAssistant: Sure."
    )
    assert "Hi?" in prompt
    assert resp == "Sure."


def test_load_fixture_randomizes_slots() -> None:
    pairs = load_fixture(FIXTURE, n_samples=8, seed=42)
    assert len(pairs) == 8
    assert {p.label for p in pairs} <= {"A", "B"}
    assert any(p.label == "A" for p in pairs)
    assert any(p.label == "B" for p in pairs)


def test_position_swap_flips_label() -> None:
    pairs = load_fixture(FIXTURE, n_samples=1, seed=0)
    swapped = position_swap(pairs[0])
    assert swapped.response_a == pairs[0].response_b
    assert swapped.label != pairs[0].label


def test_parse_verdict() -> None:
    ok = parse_verdict("Reasoning...\nVerdict: B\n", r"Verdict:\s*([AB])")
    assert ok.verdict == "B" and not ok.parse_error
    json_ok = parse_verdict('{"verdict": "A"}', r"Verdict:\s*([AB])")
    assert json_ok.verdict == "A" and not json_ok.parse_error
    last_line = parse_verdict("I pick\nA", r"Verdict:\s*([AB])")
    assert last_line.verdict == "A" and not last_line.parse_error
    bad = parse_verdict("I like the first one.", r"Verdict:\s*([AB])")
    assert bad.parse_error and bad.verdict is None


def test_wilson_and_kappa() -> None:
    lo, hi = wilson_interval(7, 10)
    assert 0 <= lo <= 0.7 <= hi <= 1
    assert cohens_kappa(["a", "a", "b"], ["a", "b", "b"]) < 1


def test_dry_run_emits_verdict() -> None:
    text = DryRunClient().complete(
        "You are a judge.",
        "Response A:\nshort\nResponse B:\na much longer reply here\nWhich response",
    )
    assert text.startswith("Verdict: ")
