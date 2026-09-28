"""Fetch and cache the default HH-RLHF subset (no model calls)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from preference_consistency.data import load_pairs
from preference_consistency.run_eval import repo_root

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download/cache HH-RLHF subset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root = repo_root()
    path = Path(args.config) if Path(args.config).is_absolute() else root / args.config
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    n_samples = args.n_samples if args.n_samples is not None else int(cfg["n_samples"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])

    pairs, origin = load_pairs(
        source="download",
        fixture_path=root / data_cfg["fixture_path"],
        cache_dir=root / data_cfg["cache_dir"],
        n_samples=n_samples,
        n_samples_max=int(cfg["n_samples_max"]),
        seed=seed,
        hh_url=str(data_cfg["hh_url"]),
        hh_split=str(data_cfg["hh_split"]),
        allow_download=True,
    )
    print(f"Ready: {len(pairs)} pairs from {origin}")


if __name__ == "__main__":
    main()
