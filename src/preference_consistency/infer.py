"""Ollama HTTP client, preflight, and verdict parsing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["A", "B"]}},
    "required": ["verdict"],
}


@dataclass
class Completion:
    text: str
    verdict: str | None
    parse_error: bool


class JudgeClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OllamaClient:
    """Local judge via Ollama `/api/chat` (instruct models)."""

    def __init__(
        self,
        host: str,
        model: str,
        temperature: float,
        seed: int,
        timeout_s: float,
        num_predict: int = 64,
        use_json_schema: bool = True,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout_s = timeout_s
        self.num_predict = num_predict
        self.use_json_schema = use_json_schema
        self._client = httpx.Client(timeout=timeout_s)
        self._schema_ok = use_json_schema

    def close(self) -> None:
        self._client.close()

    def complete(self, system: str, user: str) -> str:
        try:
            return self._chat(system, user, use_schema=self._schema_ok)
        except httpx.HTTPStatusError as exc:
            if self._schema_ok and exc.response.status_code in {400, 422, 500}:
                logger.warning("Ollama rejected JSON schema (%s); retrying unconstrained.", exc)
                self._schema_ok = False
                return self._chat(system, user, use_schema=False)
            raise

    def _chat(self, system: str, user: str, *, use_schema: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.num_predict,
            },
        }
        if use_schema:
            payload["format"] = VERDICT_JSON_SCHEMA
        try:
            resp = self._client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(offline_message(self.host, self.model)) from exc
        data = resp.json()
        content = str((data.get("message") or {}).get("content") or data.get("response") or "")
        if use_schema:
            verdict = _verdict_from_json_blob(content)
            if verdict:
                return f"Verdict: {verdict}\n{content.strip()}"
        return content


def list_ollama_models(host: str, timeout_s: float = 5.0) -> list[str]:
    url = f"{host.rstrip('/')}/api/tags"
    try:
        resp = httpx.get(url, timeout=timeout_s)
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(offline_message(host, model=None)) from exc
    models = resp.json().get("models") or []
    names: list[str] = []
    for item in models:
        name = str(item.get("name") or "")
        if name:
            names.append(name)
    return names


def model_is_present(names: list[str], wanted: str) -> bool:
    wanted = wanted.strip()
    for name in names:
        if name == wanted or name.startswith(wanted):
            return True
    return False


def preflight_ollama(host: str, model: str, timeout_s: float = 5.0) -> list[str]:
    names = list_ollama_models(host, timeout_s=timeout_s)
    if not model_is_present(names, model):
        raise RuntimeError(
            f"Ollama is running but model {model!r} is not installed.\n"
            f"Installed: {names or '(none)'}\n"
            f"Pull the default judge (about 2 GB):\n  ollama pull {model}"
        )
    return names


def offline_message(host: str, model: str | None) -> str:
    pull = f" and `ollama pull {model}`" if model else ""
    return (
        f"Cannot reach Ollama at {host}. Install it (`brew install ollama` or "
        f"https://ollama.com), start the app or `ollama serve`{pull}. "
        "Then `make doctor`."
    )


class DryRunClient:
    """Deterministic fake judge so the harness runs without a local model.

    Baseline/paraphrase/thorough: prefer the longer reply (proxy for 'quality' on
    the synthetic fixture). Position swap: always pick the first slot (bias).
    Conflict-short: prefer the shorter reply.
    """

    def complete(self, system: str, user: str) -> str:
        a = _extract_block(user, "Response A:")
        b = _extract_block(user, "Response B:")
        extra = system.lower()
        if "prefer the shorter" in extra:
            letter = "A" if len(a) <= len(b) else "B"
        elif "prefer the longer" in extra:
            letter = "A" if len(a) >= len(b) else "B"
        elif "[condition=position_swap]" in user:
            letter = "A"
        else:
            letter = "A" if len(a) >= len(b) else "B"
        return f"Verdict: {letter}"

    def close(self) -> None:
        return None


def _extract_block(user: str, header: str) -> str:
    idx = user.find(header)
    if idx < 0:
        return ""
    rest = user[idx + len(header) :]
    next_headers = ["\nResponse B:", "\nWhich response"]
    end = len(rest)
    for h in next_headers:
        found = rest.find(h)
        if found >= 0:
            end = min(end, found)
    return rest[:end].strip()


def _verdict_from_json_blob(raw: str) -> str | None:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", text, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        value = str(data.get("verdict", "")).strip().upper()
        if value in {"A", "B"}:
            return value
    return None


def parse_verdict(raw: str, pattern: str) -> Completion:
    json_verdict = _verdict_from_json_blob(raw)
    if json_verdict:
        return Completion(text=raw, verdict=json_verdict, parse_error=False)

    match = re.search(pattern, raw, flags=re.IGNORECASE)
    if match:
        return Completion(text=raw, verdict=match.group(1).upper(), parse_error=False)

    # Last non-empty line is exactly A or B (common 3B failure mode)
    lines = [ln.strip().strip("*`\"") for ln in raw.strip().splitlines() if ln.strip()]
    if lines and re.fullmatch(r"[AB]", lines[-1], flags=re.IGNORECASE):
        return Completion(text=raw, verdict=lines[-1].upper(), parse_error=False)

    logger.error("Unparseable judge output:\n%s", raw)
    return Completion(text=raw, verdict=None, parse_error=True)


def require_verdict(raw: str, pattern: str) -> Completion:
    completion = parse_verdict(raw, pattern)
    if completion.parse_error:
        logger.error("Failed to parse Verdict: A/B from model output (logged above).")
    return completion
