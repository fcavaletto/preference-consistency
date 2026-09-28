"""Load and subsample preference pairs."""

from __future__ import annotations

import gzip
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import httpx

logger = logging.getLogger(__name__)

Label = Literal["A", "B"]
ASSISTANT_MARKERS = ("\n\nAssistant:", "\nAssistant:")


@dataclass
class PreferencePair:
    id: str
    prompt: str
    response_a: str
    response_b: str
    label: Label
    chosen: str
    rejected: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_prompt_and_response(transcript: str) -> tuple[str, str]:
    """Split an HH-style Human/Assistant transcript into prompt vs last reply."""
    text = transcript.strip()
    idx = -1
    marker_used = ""
    for marker in ASSISTANT_MARKERS:
        found = text.rfind(marker)
        if found > idx:
            idx = found
            marker_used = marker
    if idx < 0:
        return "", text
    prompt = text[:idx].strip()
    prompt = re.sub(r"^Human:\s*", "", prompt, count=1)
    response = text[idx + len(marker_used) :].strip()
    return prompt, response


def pair_from_hh_row(row: dict[str, Any], rng: random.Random, pair_id: str) -> PreferencePair:
    chosen_full = str(row["chosen"])
    rejected_full = str(row["rejected"])
    prompt_c, chosen_resp = split_prompt_and_response(chosen_full)
    prompt_r, rejected_resp = split_prompt_and_response(rejected_full)
    prompt = prompt_c or prompt_r
    chosen_first = rng.random() < 0.5
    if chosen_first:
        response_a, response_b, label = chosen_resp, rejected_resp, "A"
    else:
        response_a, response_b, label = rejected_resp, chosen_resp, "B"
    return PreferencePair(
        id=pair_id,
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        label=label,  # type: ignore[arg-type]
        chosen=chosen_resp,
        rejected=rejected_resp,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_fixture(path: Path, n_samples: int, seed: int) -> list[PreferencePair]:
    rng = random.Random(seed)
    raw = _read_jsonl(path)
    if n_samples < len(raw):
        raw = rng.sample(raw, n_samples)
    pairs = []
    for i, row in enumerate(raw):
        pair_id = str(row.get("id", f"fixture-{i:04d}"))
        pairs.append(pair_from_hh_row(row, rng, pair_id))
    return pairs


def _iter_hh_gz(url: str, timeout_s: float) -> Iterator[dict[str, Any]]:
    logger.info(
        "Downloading Anthropic HH-RLHF from %s (tens of MB; one-time if you keep the cache)…",
        url,
    )
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        raw = gzip.decompress(resp.content)
    for line in raw.splitlines():
        if line.strip():
            yield json.loads(line)


def download_hh_subset(
    url: str,
    n_samples: int,
    seed: int,
    cache_path: Path,
    timeout_s: float = 180.0,
) -> list[PreferencePair]:
    """Download HH JSONL.gz, subsample with a fixed seed, cache normalized pairs."""
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for i, row in enumerate(_iter_hh_gz(url, timeout_s)):
        if i < n_samples:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < n_samples:
                reservoir[j] = row
    assign_rng = random.Random(seed)
    pairs = [
        pair_from_hh_row(row, assign_rng, f"hh-{i:05d}")
        for i, row in enumerate(reservoir)
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")
    logger.info("Cached %s pairs → %s", len(pairs), cache_path)
    return pairs


def load_cached_pairs(path: Path) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for row in _read_jsonl(path):
        pairs.append(
            PreferencePair(
                id=str(row["id"]),
                prompt=str(row["prompt"]),
                response_a=str(row["response_a"]),
                response_b=str(row["response_b"]),
                label=row["label"],
                chosen=str(row["chosen"]),
                rejected=str(row["rejected"]),
            )
        )
    return pairs


def cache_path_for(cache_dir: Path, hh_split: str, seed: int, n_samples: int) -> Path:
    return cache_dir / f"hh_subset_{hh_split}_seed{seed}_n{n_samples}.jsonl"


def load_pairs(
    *,
    source: str,
    fixture_path: Path,
    cache_dir: Path,
    n_samples: int,
    n_samples_max: int,
    seed: int,
    hh_url: str,
    hh_split: str,
    allow_download: bool | None = None,
) -> tuple[list[PreferencePair], str]:
    if n_samples > n_samples_max:
        raise ValueError(f"n_samples={n_samples} exceeds n_samples_max={n_samples_max}")
    cache_path = cache_path_for(cache_dir, hh_split, seed, n_samples)

    if source == "fixture":
        return load_fixture(fixture_path, n_samples, seed), "fixture"

    if cache_path.exists():
        logger.info("Loading HH subset from cache %s", cache_path)
        pairs = load_cached_pairs(cache_path)
        return pairs[:n_samples], str(cache_path)

    if source == "cache":
        raise FileNotFoundError(
            f"No cache at {cache_path}. Run `make fetch-data` or set data.source: download."
        )

    # Default path: download when configured as download (or caller opts in).
    if allow_download is None:
        allow_download = source == "download"

    if source == "download" or allow_download:
        if not allow_download and source != "download":
            raise RuntimeError(
                "Refusing to download HH-RLHF without download enabled "
                "(the gzip is tens of MB)."
            )
        pairs = download_hh_subset(hh_url, n_samples, seed, cache_path)
        return pairs, str(cache_path)

    raise ValueError(f"Unknown data source: {source}")
