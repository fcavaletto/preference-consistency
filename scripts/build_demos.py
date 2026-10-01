"""Rebuild results/qwen2.5-7b/demos.json from saved judgments and the HH cache.

No extra model calls: raw completions are already in judgments.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from preference_consistency.data import load_pairs
from preference_consistency.prompts import load_text, render_system, render_user
from preference_consistency.run_eval import prepare_condition, repo_root


def build(judgments_path: Path, out_path: Path, config_path: Path | None = None) -> Path:
    root = repo_root()
    cfg = yaml.safe_load((config_path or root / "configs/default.yaml").read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    pairs, _origin = load_pairs(
        source="cache",
        fixture_path=root / data_cfg["fixture_path"],
        cache_dir=root / data_cfg["cache_dir"],
        n_samples=100,
        n_samples_max=int(cfg["n_samples_max"]),
        seed=int(cfg["seed"]),
        hh_url=str(data_cfg["hh_url"]),
        hh_split=str(data_cfg["hh_split"]),
        allow_download=False,
    )
    by_id = {pair.id: pair for pair in pairs}
    rows = [json.loads(line) for line in judgments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = [
        "baseline",
        "position_swap",
        "verbosity_bloat_a",
        "verbosity_bloat_b",
        "sycophancy_prefer_a",
        "sycophancy_user_prefers_a",
    ]
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row["condition"] not in wanted:
            continue
        grouped.setdefault(row["pair_id"], {})[row["condition"]] = row
    pair_id = next(pid for pid, conds in grouped.items() if set(wanted) <= set(conds) and pid in by_id)
    pair = by_id[pair_id]
    cond_cfg = cfg["conditions"]
    system_base = load_text(root / cfg["prompts"]["system_path"])
    user_tmpl = load_text(root / cfg["prompts"]["user_path"])
    conditions = {}
    for name in wanted:
        work, extra_system, extra_user = prepare_condition(
            pair,
            name,
            root=root,
            conflict_files=cond_cfg.get("conflicting_instructions") or {},
            sycophancy_files=cond_cfg.get("sycophancy") or {},
            sycophancy_user_files=cond_cfg.get("sycophancy_user") or {},
        )
        row = grouped[pair_id][name]
        conditions[name] = {
            "system": render_system(system_base, extra_system),
            "user": render_user(user_tmpl, work, extra=extra_user),
            "raw": row.get("raw"),
            "verdict": row.get("verdict"),
            "winner_content": row.get("winner_content"),
            "correct": row.get("correct"),
            "parse_error": row.get("parse_error"),
        }
    payload = {
        "model": "qwen2.5:7b",
        "pair": pair.as_dict(),
        "conditions": conditions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    root = repo_root()
    path = build(
        root / "results/qwen2.5-7b/judgments.jsonl",
        root / "results/qwen2.5-7b/demos.json",
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
