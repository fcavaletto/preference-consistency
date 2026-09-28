"""Load judge templates from files."""

from __future__ import annotations

from pathlib import Path

from preference_consistency.data import PreferencePair


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def render_user(template: str, pair: PreferencePair, extra: str = "") -> str:
    body = template.format(
        prompt=pair.prompt,
        response_a=pair.response_a,
        response_b=pair.response_b,
    )
    if extra:
        return f"{extra.rstrip()}\n\n{body}"
    return body


def render_system(base: str, extra: str = "") -> str:
    if extra:
        return f"{base.rstrip()}\n\n{extra.strip()}"
    return base
