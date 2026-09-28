"""Input perturbations: position swap, rule-based paraphrases, extra instructions."""

from __future__ import annotations

import re
from dataclasses import replace

from preference_consistency.data import PreferencePair


def position_swap(pair: PreferencePair) -> PreferencePair:
    new_label = "B" if pair.label == "A" else "A"
    return replace(
        pair,
        response_a=pair.response_b,
        response_b=pair.response_a,
        label=new_label,  # type: ignore[arg-type]
    )


def paraphrase_whitespace(text: str) -> str:
    collapsed = re.sub(r"[ \t]+", " ", text)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip() + "\n"


def paraphrase_role_markup(text: str) -> str:
    text = re.sub(r"\bHuman:\s*", "User: ", text)
    text = re.sub(r"\bAssistant:\s*", "AI: ", text)
    return text.strip()


def paraphrase_lexical(text: str) -> str:
    prefix = "Please help with the following request.\n\n"
    if text.lower().startswith("please help"):
        return text
    return prefix + text


PARAPHRASERS = {
    "whitespace": paraphrase_whitespace,
    "role_markup": paraphrase_role_markup,
    "lexical": paraphrase_lexical,
}


def apply_paraphrase(pair: PreferencePair, name: str) -> PreferencePair:
    fn = PARAPHRASERS[name]
    return replace(
        pair,
        prompt=fn(pair.prompt),
        response_a=fn(pair.response_a),
        response_b=fn(pair.response_b),
    )
