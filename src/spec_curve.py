"""Specification-curve figure, shared by the main and supplementary multiverse figures."""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from src import plot_style as ps
from src.config import ALPHA
from src.ersp import BETA_FMAX, BETA_FMIN

CI_LEVEL = 95
SIG_COLOR = "#018532"  # comet's significance green
# CI band as in comet's Multiverse.specification_curve (smooth_ci=True): cubic
# interpolating splines through the per-universe bounds, 500 points, grey, alpha 0.3.
BAND_POINTS = 500
BAND_ALPHA = 0.3


def ci_band(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed CI band between universes (comet's smoothing).

    Parameters
    ----------
    table : pandas.DataFrame
        One row per universe, sorted by ``mean``, with ``ci_lo`` and ``ci_hi``.

    Returns
    -------
    x, lower, upper : numpy.ndarray
        Band abscissa (universe index units) and its lower and upper edge, dB.
        The edges pass through the stored bounds at every universe.
    """
    x = np.arange(len(table), dtype=float)
    spline_lo = make_interp_spline(x, table["ci_lo"].to_numpy(float), k=3)
    spline_hi = make_interp_spline(x, table["ci_hi"].to_numpy(float), k=3)
    x_band = np.linspace(x.min(), x.max(), BAND_POINTS)
    return x_band, spline_lo(x_band), spline_hi(x_band)


def band_report(table: pd.DataFrame) -> None:
    """Print how the band matches the stored CIs at each universe and between universes."""
    x = np.arange(len(table), dtype=float)
    lo, hi = table["ci_lo"].to_numpy(float), table["ci_hi"].to_numpy(float)
    spline_lo = make_interp_spline(x, lo, k=3)
    spline_hi = make_interp_spline(x, hi, k=3)
    at = max(np.abs(spline_lo(x) - lo).max(), np.abs(spline_hi(x) - hi).max())
    print(f"  CI band at universes: max |band - stored CI| = {at:.2e} dB")
    x_band, band_lo, band_hi = ci_band(table)
    seg = np.minimum(np.floor(x_band).astype(int), len(x) - 2)
    below = np.minimum(lo[seg], lo[seg + 1]) - band_lo
    above = band_hi - np.maximum(hi[seg], hi[seg + 1])
    print(
        f"  Between universes: lower edge dips up to {max(below.max(), 0):.3f} dB below the lower "
        f"of the two neighbouring bounds ({int((below > 1e-9).sum())} of {len(x_band)} points); "
        f"upper edge rises up to {max(above.max(), 0):.3f} dB above the higher one"
    )


def check_ci_within_ylim(table: pd.DataFrame, ylim: tuple[float, float]) -> None:
    """Raise SystemExit if any confidence interval falls outside the fixed y-limits.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``src.multiverse_results.universe_table``.
    ylim : tuple of float
        Fixed (lower, upper) y-limits of the estimate panel, dB.
    """
    _, band_lo, band_hi = ci_band(table)
    lowest = min(table["ci_lo"].min(), band_lo.min())
    highest = max(table["ci_hi"].max(), band_hi.max())
    if lowest < ylim[0] or highest > ylim[1]:
        raise SystemExit("A confidence interval (or its band) falls outside the fixed y-limits.")


def draw_estimates(ax: plt.Axes, table: pd.DataFrame, ylim: tuple[float, float]) -> None:
    """Draw per-universe means, the smoothed CI band, the zero line and the legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Upper panel.
    table : pandas.DataFrame
        One row per universe, sorted by ``mean``.
    ylim : tuple of float
        Fixed y-limits, dB.
    """
    significant = table["p"] < ALPHA
    x = np.arange(len(table))
    colors = np.where(significant, SIG_COLOR, "black")
    x_band, band_lo, band_hi = ci_band(table)
    ax.fill_between(x_band, band_lo, band_hi, color="grey", alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.scatter(x, table["mean"], c=colors, s=36, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylim(*ylim)
    ax.set_xlim(-1, len(table))
    ax.set_ylabel(ps.SPEC_YLABEL, fontsize=ps.FS_LABEL)
    ax.grid(axis="y", color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    label = f"{CI_LEVEL}% CI (t-based, smoothed between universes)"
    legend = [mpatches.Patch(facecolor="grey", alpha=BAND_ALPHA, linewidth=0, label=label)]
    for show, color, text in (
        (significant.any(), SIG_COLOR, f"p < {ALPHA}, one-sample t-test vs 0, uncorrected"),
        ((~significant).any(), "black", f"p >= {ALPHA}, uncorrected"),
    ):
        if show:
            marker = mlines.Line2D([], [], linestyle="None", marker="o", color=color, label=text)
            legend.append(marker)
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=ps.FS_NOTE)


def draw_decisions(ax: plt.Axes, table: pd.DataFrame) -> None:
    """Draw one row per decision option (Table 2.1 order) marking each universe's values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Lower panel, sharing the x-axis with the estimate panel.
    table : pandas.DataFrame
        One row per universe, sorted by ``mean``.
    """
    x = np.arange(len(table))
    group_colors = [plt.get_cmap("Set3")(i) for i in range(3)]
    yticks, ylabels, y = [], [], 0
    for g, (decision, options) in enumerate(ps.NODE_ORDER.items()):
        first = y
        for option in options:
            if decision == "highpass_hz":
                hits = np.asarray(np.isclose(table[decision].astype(float), option))
            else:
                hits = np.asarray(table[decision] == option)
            ax.scatter(
                x[hits],
                np.full(int(hits.sum()), y),
                color=group_colors[g],
                edgecolor="0.4",
                linewidth=0.4,
                s=36,
                zorder=3,
            )
            ax.axhline(y, color="0.9", linewidth=0.8, zorder=1)
            yticks.append(y)
            ylabels.append(ps.option_label(decision, option))
            y += 1
        ax.text(
            -0.2,
            (first + y - 1) / 2,
            ps.DECISION_LABELS[decision],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=ps.FS_LABEL,
            fontweight="bold",
        )
        y += 0.6
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=ps.FS_TICK)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xticks([])
    ax.set_xlabel("Universes, sorted by contrast", fontsize=ps.FS_LABEL)
    ax.spines[["top", "right"]].set_visible(False)


def plot_spec_curve(
    table: pd.DataFrame,
    dataset: str,
    title_content: str,
    n_label: int | str,
    ylim: tuple[float, float],
    out_path: Path,
) -> None:
    """Save a specification curve: estimates with CIs above, decision rows below.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``src.multiverse_results.universe_table`` (sorted by mean).
    dataset : str
        "stepup" or "jacobsen" (used for the "Treadmill"/"Overground" title).
    title_content : str
        Title text after "<Dataset> walking: ".
    n_label : int or str
        Subjects per universe, shown as "(n = ...)".
    ylim : tuple of float
        Fixed y-limits of the estimate panel, dB.
    out_path : pathlib.Path
        PNG path.
    """
    ps.apply_rc()
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=ps.FIGSIZE["speccurve"],
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1],
            "hspace": 0.08,
            "top": 0.9,
            "left": 0.24,
            "right": 0.97,
        },
    )
    ps.add_header(
        fig,
        ps.figure_title(dataset, title_content, n_label),
        ps.figure_subtitle(BETA_FMIN, BETA_FMAX),
    )
    band_report(table)
    draw_estimates(ax_top, table, ylim)
    draw_decisions(ax_bot, table)
    ps.save(fig, out_path)
