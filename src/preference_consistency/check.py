"""Preflight: Ollama daemon + default judge weights."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from preference_consistency.infer import list_ollama_models, model_is_present

logger = logging.getLogger(__name__)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_model_cfg(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg["model"]


def doctor(config: str = "configs/default.yaml") -> int:
    root = repo_root()
    path = Path(config) if Path(config).is_absolute() else root / config
    model_cfg = load_model_cfg(path)
    host = str(model_cfg["host"])
    model = str(model_cfg["name"])
    try:
        names = list_ollama_models(host)
    except RuntimeError as exc:
        print(exc)
        print("Fix: brew install ollama && ollama serve   # or open the Ollama app")
        print(f"Then: ollama pull {model}")
        return 2
    print(f"Ollama OK at {host}")
    print(f"Installed models: {', '.join(names) if names else '(none)'}")
    if not model_is_present(names, model):
        print(f"Missing default judge {model!r}.")
        print(f"  ollama pull {model}")
        return 3
    print(f"Default judge ready: {model}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check local Ollama judge")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(doctor(args.config))


if __name__ == "__main__":
    main()
