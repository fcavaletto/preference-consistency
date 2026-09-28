"""CLI: run preference-consistency conditions and write results."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from preference_consistency.data import PreferencePair, load_pairs
from preference_consistency.infer import (
    DryRunClient,
    JudgeClient,
    OllamaClient,
    preflight_ollama,
    require_verdict,
)
from preference_consistency.metrics import Rate, agreement, cohens_kappa, fmt_rate, rate
from preference_consistency.perturbations import apply_paraphrase, position_swap
from preference_consistency.prompts import load_text, render_system, render_user

logger = logging.getLogger(__name__)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def winner_side(pair: PreferencePair, letter: str) -> str:
    """Map verdict letter to chosen/rejected content identity."""
    if letter == "A":
        return "chosen" if pair.label == "A" else "rejected"
    if letter == "B":
        return "chosen" if pair.label == "B" else "rejected"
    raise ValueError(letter)


def picked_first(letter: str) -> bool:
    return letter == "A"


def judge_one(
    client: JudgeClient,
    pair: PreferencePair,
    *,
    system: str,
    user_template: str,
    verdict_pattern: str,
    condition: str,
    extra_user: str,
    dry_run: bool,
) -> dict[str, Any]:
    user = render_user(user_template, pair, extra=extra_user)
    if dry_run:
        user = f"{user}\n\n[condition={condition}]"
    raw = client.complete(system, user)
    completion = require_verdict(raw, verdict_pattern)
    record: dict[str, Any] = {
        "pair_id": pair.id,
        "condition": condition,
        "label": pair.label,
        "raw": completion.text,
        "parse_error": completion.parse_error,
        "verdict": completion.verdict,
        "winner_content": None,
        "correct": None,
        "picked_first": None,
    }
    if completion.parse_error or completion.verdict is None:
        logger.error("parse_error pair_id=%s condition=%s", pair.id, condition)
        return record
    letter = completion.verdict
    content = winner_side(pair, letter)
    record["winner_content"] = content
    record["correct"] = content == "chosen"
    record["picked_first"] = picked_first(letter)
    return record


def collect_conditions(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> list[str]:
    cond = cfg["conditions"]
    names: list[str] = []
    if cond.get("baseline", True):
        names.append("baseline")
    if cond.get("position_swap", True):
        names.append("position_swap")
    for para in cond.get("paraphrases") or []:
        names.append(f"paraphrase_{para}")
    for key in (cond.get("conflicting_instructions") or {}):
        names.append(f"conflict_{key}")
    for key in (cond.get("sycophancy") or {}):
        names.append(f"sycophancy_{key}")
    if args is not None and getattr(args, "conditions", None):
        wanted = set(args.conditions)
        names = [n for n in names if n in wanted]
    return names


def run(args: argparse.Namespace) -> Path:
    root = repo_root()
    cfg = load_config(Path(args.config) if Path(args.config).is_absolute() else root / args.config)
    n_samples = args.n_samples if args.n_samples is not None else int(cfg["n_samples"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    data_cfg = cfg["data"]
    source = "fixture" if args.fixture else data_cfg["source"]
    if args.download_hh:
        source = "download"

    pairs, data_origin = load_pairs(
        source=source,
        fixture_path=root / data_cfg["fixture_path"],
        cache_dir=root / data_cfg["cache_dir"],
        n_samples=n_samples,
        n_samples_max=int(cfg["n_samples_max"]),
        seed=seed,
        hh_url=str(data_cfg["hh_url"]),
        hh_split=str(data_cfg["hh_split"]),
        allow_download=True if source == "download" else bool(args.download_hh),
    )
    logger.info("Loaded %s pairs from %s", len(pairs), data_origin)

    prompts_cfg = cfg["prompts"]
    system_base = load_text(root / prompts_cfg["system_path"])
    user_template = load_text(root / prompts_cfg["user_path"])
    verdict_pattern = str(prompts_cfg["verdict_pattern"])
    conflict_files = cfg["conditions"].get("conflicting_instructions") or {}
    sycophancy_files = cfg["conditions"].get("sycophancy") or {}

    model_cfg = cfg["model"]
    dry_run = bool(args.dry_run)
    client: JudgeClient
    if dry_run:
        client = DryRunClient()
        model_name = "dry-run"
    else:
        host = str(model_cfg["host"])
        model_name = str(model_cfg["name"])
        preflight_ollama(host, model_name)
        client = OllamaClient(
            host=host,
            model=model_name,
            temperature=float(model_cfg["temperature"]),
            seed=int(model_cfg["seed"]),
            timeout_s=float(model_cfg["timeout_s"]),
            num_predict=int(model_cfg.get("num_predict", 64)),
            use_json_schema=bool(model_cfg.get("json_schema", True)),
        )

    condition_names = collect_conditions(cfg, args)
    logger.info("Running %s pairs × %s conditions (model=%s)", len(pairs), condition_names, model_name)

    judgments: list[dict[str, Any]] = []
    try:
        total = len(pairs) * len(condition_names)
        step = 0
        for pair in pairs:
            for condition in condition_names:
                step += 1
                work = pair
                extra_system = ""
                if condition == "position_swap":
                    work = position_swap(pair)
                elif condition.startswith("paraphrase_"):
                    work = apply_paraphrase(pair, condition.removeprefix("paraphrase_"))
                elif condition.startswith("conflict_"):
                    key = condition.removeprefix("conflict_")
                    extra_system = load_text(root / conflict_files[key])
                elif condition.startswith("sycophancy_"):
                    key = condition.removeprefix("sycophancy_")
                    extra_system = load_text(root / sycophancy_files[key])
                system = render_system(system_base, extra_system)
                logger.info("[%s/%s] %s %s", step, total, pair.id, condition)
                judgments.append(
                    judge_one(
                        client,
                        work,
                        system=system,
                        user_template=user_template,
                        verdict_pattern=verdict_pattern,
                        condition=condition,
                        extra_user="",
                        dry_run=dry_run,
                    )
                )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / str(cfg["output"]["judgments_name"])
    with judgments_path.open("w", encoding="utf-8") as handle:
        for row in judgments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sample_path = out_dir / str(cfg["output"]["sample_name"])
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in judgments[:16]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    results_md = write_results_md(
        out_dir / str(cfg["output"]["results_name"]),
        judgments=judgments,
        pairs=pairs,
        model_name=model_name,
        dry_run=dry_run,
        n_samples=len(pairs),
        seed=seed,
        source=source,
        temperature=float(model_cfg["temperature"]),
    )
    logger.info("Wrote %s and %s", judgments_path, results_md)
    return results_md


def _by_condition(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["condition"] == name].copy()


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[(~df["parse_error"]) & df["verdict"].notna()].copy()


def _paired_content(baseline: pd.DataFrame, other: pd.DataFrame) -> tuple[list[str], list[str]]:
    b = baseline.set_index("pair_id")["winner_content"]
    o = other.set_index("pair_id")["winner_content"]
    ids = sorted(set(b.index) & set(o.index))
    return [str(b.loc[i]) for i in ids], [str(o.loc[i]) for i in ids]


def write_results_md(
    path: Path,
    *,
    judgments: list[dict[str, Any]],
    pairs: list[PreferencePair],
    model_name: str,
    dry_run: bool,
    n_samples: int,
    seed: int,
    source: str,
    temperature: float,
) -> Path:
    df = pd.DataFrame(judgments)
    parse_errors = int(df["parse_error"].sum()) if not df.empty else 0
    lines: list[str] = []
    lines.append("# Preference consistency results")
    lines.append("")
    lines.append("Small-scale **prompted local judge** study. Not a trained reward model and not SOTA.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Dry-run: `{dry_run}`")
    lines.append(f"- Temperature: `{temperature}` (greedy; documented even if a paper used sampling)")
    lines.append(f"- n pairs: `{n_samples}`  seed: `{seed}`  data: `{source}`")
    lines.append(f"- Parse errors: `{parse_errors}` / `{len(df)}` judgments")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")

    def acc(frame: pd.DataFrame) -> Rate:
        v = _valid(frame)
        if v.empty:
            return rate(0, 0)
        return rate(int(v["correct"].sum()), len(v))

    def first_bias(frame: pd.DataFrame) -> Rate:
        v = _valid(frame)
        if v.empty:
            return rate(0, 0)
        return rate(int(v["picked_first"].sum()), len(v))

    by = {name: _by_condition(df, name) for name in df["condition"].unique()} if not df.empty else {}
    baseline = by.get("baseline", pd.DataFrame())
    swap = by.get("position_swap", pd.DataFrame())

    lines.append("| Condition | Accuracy vs HH label | P(pick first slot) |")
    lines.append("|---|---|---|")
    for name, frame in sorted(by.items()):
        lines.append(f"| `{name}` | {fmt_rate(acc(frame))} | {fmt_rate(first_bias(frame))} |")
    lines.append("")

    if not baseline.empty and not swap.empty:
        vb = _valid(baseline)
        vs = _valid(swap)
        left, right = _paired_content(vb, vs)
        flip = agreement(left, right)
        # flip rate = 1 - agreement on content identity
        flip_rate = rate(flip.n - flip.k, flip.n) if flip.n else rate(0, 0)
        kappa = cohens_kappa(left, right)
        lines.append("### Position swap (Zheng / Wang-style)")
        lines.append("")
        lines.append(f"- Content-level agreement after A↔B: {fmt_rate(flip)}")
        lines.append(f"- **Flip rate**: {fmt_rate(flip_rate)}")
        lines.append(f"- Cohen's κ (baseline vs swap, content id): `{kappa:.3f}`" if flip.n else "- Cohen's κ: n/a")
        lines.append("")

    para_names = [n for n in by if n.startswith("paraphrase_")]
    if baseline.empty is False and para_names:
        lines.append("### Paraphrase self-consistency")
        lines.append("")
        vb = _valid(baseline)
        for name in para_names:
            left, right = _paired_content(vb, _valid(by[name]))
            agr = agreement(left, right)
            kap = cohens_kappa(left, right)
            lines.append(f"- `{name}` vs baseline: {fmt_rate(agr)}; κ = `{kap:.3f}`")
        lines.append("")

    conflicts = [n for n in by if n.startswith("conflict_")]
    if len(conflicts) >= 2:
        a, b = conflicts[0], conflicts[1]
        left, right = _paired_content(_valid(by[a]), _valid(by[b]))
        agr = agreement(left, right)
        kap = cohens_kappa(left, right)
        lines.append("### Conflicting instructions")
        lines.append("")
        lines.append(f"- `{a}` vs `{b}` content agreement: {fmt_rate(agr)}; κ = `{kap:.3f}`")
        lines.append("")

    syc_names = sorted(n for n in by if n.startswith("sycophancy_"))
    if baseline.empty is False and syc_names:
        lines.append("### Sycophancy / judge bias (Perez-style, lightweight)")
        lines.append("")
        lines.append(
            "We bias the *judge* with an extra instruction (user prefers A/B, or “agree with the user”). "
            "This is **not** a full reproduction of Perez et al.'s model-written evaluation suite "
            "([arXiv:2212.09251](https://arxiv.org/abs/2212.09251)); it is a small protocol to ask: "
            "does the judge follow the stated preference over content?"
        )
        lines.append("")
        vb = _valid(baseline)
        for name in syc_names:
            left, right = _paired_content(vb, _valid(by[name]))
            agr = agreement(left, right)
            flip = rate(agr.n - agr.k, agr.n) if agr.n else rate(0, 0)
            kap = cohens_kappa(left, right)
            lines.append(
                f"- `{name}` vs baseline: agreement {fmt_rate(agr)}; "
                f"flip {fmt_rate(flip)}; κ = `{kap:.3f}`"
            )
            if name.endswith("prefer_a"):
                vs = _valid(by[name])
                if not vs.empty:
                    pick_a = rate(int((vs["verdict"] == "A").sum()), len(vs))
                    lines.append(f"  - P(verdict=A | prefer_a): {fmt_rate(pick_a)}")
            if name.endswith("prefer_b"):
                vs = _valid(by[name])
                if not vs.empty:
                    pick_b = rate(int((vs["verdict"] == "B").sum()), len(vs))
                    lines.append(f"  - P(verdict=B | prefer_b): {fmt_rate(pick_b)}")
        lines.append("")

    lines.append("## Comparison to papers (order of magnitude, not a copy)")
    lines.append("")
    lines.append("| Source | Reported pattern | This run |")
    lines.append("|---|---|---|")
    lines.append(
        "| [Bai et al. 2022 (HH-RLHF)](https://arxiv.org/abs/2204.05862) | Human `chosen`/`rejected` pairs | "
        "Accuracy vs those labels on a seeded subset |"
    )
    lines.append(
        "| [Zheng et al. 2023 (LLM-as-judge)](https://arxiv.org/abs/2306.05685) | Position bias; judge both orders | "
        "Flip rate / P(first) above |"
    )
    lines.append(
        "| [Wang et al. 2023 (not fair evaluators)](https://arxiv.org/abs/2305.17926) | Pairwise judges prefer a position | "
        "Same swap protocol; expect **higher** flips on 3B local judges |"
    )
    lines.append(
        "| [Perez et al. 2022 (sycophancy)](https://arxiv.org/abs/2212.09251) | Models change answers to agree with users | "
        "Judge-side bias prompts; see sycophancy section |"
    )
    lines.append("")
    lines.append("### Why numbers will differ from the papers")
    lines.append("")
    lines.append("- **Prompted 3B chat model ≠ trained Bradley–Terry RM** or GPT-4-as-judge.")
    lines.append("- **n is tens–hundreds**, not MT-Bench / Arena scale; CIs will be wide.")
    lines.append("- **HH helpfulness pairs** are not MT-Bench multi-turn scoring items.")
    lines.append("- **Temperature 0** local decoding vs API sampling in some judge papers.")
    lines.append("- **Rule-based paraphrases** understate LLM paraphrase sensitivity.")
    lines.append("- **Sycophancy conditions** bias the judge prompt; they are not Perez's full dataset pipeline.")
    if dry_run:
        lines.append("- **This file was produced with `--dry-run`**. Heuristic rules, not a model.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("See the README. Do not cite these figures as a copy of GPT-4 or Anthropic RM results.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local LLM-as-judge preference consistency eval")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--fixture", action="store_true", help="Use bundled tiny JSONL (no download)")
    p.add_argument("--dry-run", action="store_true", help="Deterministic fake judge (no Ollama)")
    p.add_argument(
        "--download-hh",
        action="store_true",
        help="Force HH download source (default config already uses download+cache)",
    )
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 — CLI should fail loudly
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
