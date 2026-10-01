"""Matplotlib figures for a preference-consistency study.

Figures use Wilson intervals already computed in `metrics`. The notebook and
`python -m preference_consistency.plots` share these functions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from preference_consistency.metrics import Rate, dual_order_consistency, rate, slot_attraction

BLUE = "#1F4E79"
AMBER = "#C47B2B"
INK = "#1A1A1A"


def load_judgments(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def valid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.loc[~df["parse_error"].astype(bool) & df["verdict"].notna()].copy()
    return frame


def _by(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty or "condition" not in df.columns:
        return df.iloc[0:0]
    return df.loc[df["condition"] == name]


def accuracy_rate(frame: pd.DataFrame) -> Rate:
    v = valid(frame)
    if v.empty:
        return rate(0, 0)
    return rate(int(v["correct"].astype(bool).sum()), len(v))


def letter_rate(frame: pd.DataFrame, letter: str) -> Rate:
    v = valid(frame)
    if v.empty:
        return rate(0, 0)
    return rate(int((v["verdict"] == letter).sum()), len(v))


def position_stats(df: pd.DataFrame) -> dict[str, Rate | float | int] | None:
    base = valid(_by(df, "baseline")).set_index("pair_id")
    swap = valid(_by(df, "position_swap")).set_index("pair_id")
    if base.empty or swap.empty:
        return None
    ids = sorted(set(base.index) & set(swap.index))
    if not ids:
        return None
    left = [str(base.loc[i, "winner_content"]) for i in ids]
    right = [str(swap.loc[i, "winner_content"]) for i in ids]
    correct = [bool(base.loc[i, "correct"]) for i in ids]
    return dual_order_consistency(left, right, correct)


def _aligned_verdicts(df: pd.DataFrame, left_name: str, right_name: str) -> tuple[list[str], list[str]]:
    left = valid(_by(df, left_name)).set_index("pair_id")["verdict"]
    right = valid(_by(df, right_name)).set_index("pair_id")["verdict"]
    ids = sorted(set(left.index) & set(right.index))
    return [str(left.loc[i]) for i in ids], [str(right.loc[i]) for i in ids]


def highlights(df: pd.DataFrame) -> list[tuple[str, Rate]]:
    """A short comparable set of rates for one study."""
    rows: list[tuple[str, Rate]] = []
    base = _by(df, "baseline")
    if not valid(base).empty:
        rows.append(("Accuracy vs HH", accuracy_rate(base)))
        rows.append(("P(pick A)", letter_rate(base, "A")))
    stats = position_stats(df)
    if stats is not None:
        flip = stats["flip"]
        if isinstance(flip, Rate):
            rows.append(("Position flip rate", flip))
    if not _by(df, "verbosity_bloat_a").empty:
        base_letters, bloated = _aligned_verdicts(df, "baseline", "verbosity_bloat_a")
        if base_letters:
            rows.append(("Attraction onto bloated A", slot_attraction(base_letters, bloated, "A")))
    if not _by(df, "sycophancy_user_prefers_a").empty:
        rows.append(("P(A | user prefers A)", letter_rate(_by(df, "sycophancy_user_prefers_a"), "A")))
    if not _by(df, "sycophancy_prefer_a").empty:
        rows.append(("P(A | system prefers A)", letter_rate(_by(df, "sycophancy_prefer_a"), "A")))
    return rows


def _yerr(rates: list[Rate]) -> list[list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for item in rates:
        if item.n <= 0:
            lower.append(0.0)
            upper.append(0.0)
            continue
        lower.append(max(0.0, item.point - item.ci_low))
        upper.append(max(0.0, item.ci_high - item.point))
    return [lower, upper]


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#B0B0B0")
    ax.set_axisbelow(True)


def figure_condition_rates(df: pd.DataFrame, metric: str, title: str) -> plt.Figure:
    """Bar chart of accuracy or P(pick A) for every condition, with Wilson bars."""
    names = sorted(df["condition"].unique())
    rates: list[Rate] = []
    labels: list[str] = []
    for name in names:
        frame = _by(df, name)
        item = accuracy_rate(frame) if metric == "accuracy" else letter_rate(frame, "A")
        if item.n <= 0:
            continue
        labels.append(name)
        rates.append(item)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    xs = range(len(labels))
    ax.bar(list(xs), [r.point for r in rates], color=BLUE, width=0.72, zorder=2)
    ax.errorbar(
        list(xs),
        [r.point for r in rates],
        yerr=_yerr(rates),
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(list(xs), labels, rotation=35, ha="right")
    ylabel = "Accuracy vs human chosen" if metric == "accuracy" else "P(verdict = A)"
    _style(ax, title, ylabel)
    fig.tight_layout()
    return fig


def figure_verbosity(df: pd.DataFrame, title: str) -> plt.Figure | None:
    if _by(df, "verbosity_bloat_a").empty or _by(df, "verbosity_bloat_b").empty:
        return None
    labels = ["P(A) baseline", "P(A) after bloat A", "P(B) baseline", "P(B) after bloat B"]
    rates = [
        letter_rate(_by(df, "baseline"), "A"),
        letter_rate(_by(df, "verbosity_bloat_a"), "A"),
        letter_rate(_by(df, "baseline"), "B"),
        letter_rate(_by(df, "verbosity_bloat_b"), "B"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = [BLUE, AMBER, BLUE, AMBER]
    ax.bar(range(4), [r.point for r in rates], color=colors, width=0.72, zorder=2)
    ax.errorbar(
        range(4),
        [r.point for r in rates],
        yerr=_yerr(rates),
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(range(4), labels, rotation=15, ha="right")
    _style(ax, title, "Probability")
    fig.tight_layout()
    return fig


def figure_scale(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_label: str,
    right_label: str,
    title: str,
) -> plt.Figure | None:
    """Grouped bars for highlights present in both studies."""
    a = dict(highlights(left))
    b = dict(highlights(right))
    labels = [name for name in a if name in b]
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    xs = list(range(len(labels)))
    width = 0.36
    ra = [a[name] for name in labels]
    rb = [b[name] for name in labels]
    ax.bar([x - width / 2 for x in xs], [r.point for r in ra], width=width, color=AMBER, label=left_label, zorder=2)
    ax.bar([x + width / 2 for x in xs], [r.point for r in rb], width=width, color=BLUE, label=right_label, zorder=2)
    ax.errorbar(
        [x - width / 2 for x in xs],
        [r.point for r in ra],
        yerr=_yerr(ra),
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
        zorder=3,
    )
    ax.errorbar(
        [x + width / 2 for x in xs],
        [r.point for r in rb],
        yerr=_yerr(rb),
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    _style(ax, title, "Rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_figures(
    judgments: Path,
    out_dir: Path,
    *,
    title_prefix: str,
    compare: Path | None = None,
    compare_label: str = "llama3.2:3b",
) -> list[Path]:
    df = load_judgments(judgments)
    written: list[Path] = []
    acc = figure_condition_rates(df, "accuracy", f"{title_prefix}: accuracy vs HH label")
    written.append(save_figure(acc, out_dir / "accuracy.png"))
    first = figure_condition_rates(df, "first", f"{title_prefix}: P(verdict = A)")
    written.append(save_figure(first, out_dir / "pick_first.png"))
    verbose = figure_verbosity(df, f"{title_prefix}: does bloat pull the verdict?")
    if verbose is not None:
        written.append(save_figure(verbose, out_dir / "verbosity.png"))
    if compare is not None and compare.exists():
        other = load_judgments(compare)
        scaled = figure_scale(
            other,
            df,
            compare_label,
            title_prefix,
            "Same protocol, two model sizes",
        )
        if scaled is not None:
            written.append(save_figure(scaled, out_dir / "scale_comparison.png"))
    return written


def main(argv: list[str] | None = None) -> None:
    matplotlib.use("Agg", force=True)
    parser = argparse.ArgumentParser(description="Write preference-consistency figures")
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="qwen2.5:7b")
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--compare-label", default="llama3.2:3b")
    args = parser.parse_args(argv)
    paths = write_figures(
        args.judgments,
        args.out,
        title_prefix=args.title,
        compare=args.compare,
        compare_label=args.compare_label,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
