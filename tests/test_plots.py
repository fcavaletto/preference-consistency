"""Figure helpers on a tiny in-memory judgment table."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from preference_consistency.plots import figure_condition_rates, highlights, position_stats


def _rows() -> list[dict]:
    rows = []
    for i in range(8):
        letter = "A" if i < 5 else "B"
        rows.append(
            {
                "pair_id": f"p{i}",
                "condition": "baseline",
                "verdict": letter,
                "winner_content": "chosen" if letter == "A" else "rejected",
                "correct": letter == "A",
                "parse_error": False,
                "picked_first": letter == "A",
            }
        )
        # Swap flips content on the last two pairs only.
        swap_letter = "B" if letter == "A" else "A"
        if i >= 6:
            swap_content = "chosen" if rows[-1]["winner_content"] == "rejected" else "rejected"
        else:
            swap_content = rows[-1]["winner_content"]
            swap_letter = "B" if letter == "A" else "A"
        rows.append(
            {
                "pair_id": f"p{i}",
                "condition": "position_swap",
                "verdict": swap_letter,
                "winner_content": swap_content,
                "correct": swap_content == "chosen",
                "parse_error": False,
                "picked_first": swap_letter == "A",
            }
        )
    return rows


def test_highlights_include_accuracy_and_flip() -> None:
    import pandas as pd

    df = pd.DataFrame(_rows())
    names = [name for name, _rate in highlights(df)]
    assert "Accuracy vs HH" in names
    assert "Position flip rate" in names
    stats = position_stats(df)
    assert stats is not None
    assert stats["flip"].k == 2
    fig = figure_condition_rates(df, "accuracy", "test")
    assert fig.axes
    import matplotlib.pyplot as plt

    plt.close(fig)
