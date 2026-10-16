"""
Shared style for every results figure (main + supplementary).

Every figure script imports its colormap, colour range, fonts, figure
size, dpi, titles, label maps, decision-node order and the common
gait-cycle drawing helpers (heatmap, beta trace, event row, phase bar)
from here -- no figure script keeps its own copy.

Colour convention (ERSP heatmaps and beta topographies, both datasets):
RdBu_r, zero-centred, white = 0 dB = standing baseline, blue = below
baseline (ERD), red = above baseline (ERS). Ranges are fixed constants
below (ERSP_VLIM), one per dataset x figure type, chosen by the rule:
the smallest symmetric limit in VLIM_CANDIDATES that leaves at most
MAX_CLIP of that figure's pixels clipped. Scripts print the clipped
fraction on every run (report_clipping) so a drift is visible.

Display only: nothing here changes any computed value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from src.config import (
    DATASET_LABELS,
    DECISION_LABELS,  # noqa: F401  (re-exported for figure scripts)
    MULTIVERSE_GRID,
    OPTION_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.gridspec import SubplotSpec
    from matplotlib.image import AxesImage

# --- Colormaps ---------------------------------------------------------
CMAP = "RdBu_r"  # ERSP heatmaps, beta topographies, GPM
WEIGHT_CMAP = "Reds"  # ROI weights (non-negative by construction)

# --- Fonts / output ------------------------------------------------------
FS_TITLE, FS_LABEL, FS_TICK, FS_NOTE = 12, 11, 9, 8
DPI = 300

FIGSIZE = {
    "reference": (14, 10),
    "zoom": (16, 10),
    "speccurve": (11, 8),
    "gpm": (16, 10),
    "roi_topo": (5.5, 5.5),
}

# --- Colour ranges (dB, symmetric) ---------------------------------------
VLIM_CANDIDATES = (2, 2.5, 3, 3.5, 4, 5, 6)
MAX_CLIP = 0.02
ERSP_VLIM = {
    ("stepup", "reference"): 3.0,
    ("jacobsen", "reference"): 3.0,
    ("stepup", "zoom"): 5.0,
    ("jacobsen", "zoom"): 3.5,
}
# Reference-figure topographies have their own colour bar. Same rule over the
# 3 x 64 channel values of each figure: overground +/-3 dB (1.04 % clipped).
# Treadmill: no candidate clips <= 2 %; +/-6 dB, the largest candidate, clips
# 2.60 % (5 of 192 values, double stance, Fz/Fpz/F1/F2/AF3, max +7.76 dB).
TOPO_VLIM = {"stepup": 6.0, "jacobsen": 3.0}
TOPO_CBAR_LABEL = "Beta ERSP (dB re standing baseline)"
# GPM spans only about +-0.6 dB, below the smallest ERSP candidate, so the
# same rule is applied over a finer candidate set.
GPM_VLIM_CANDIDATES = (0.5, 1.0, 1.5, 2.0)
GPM_VLIM = 1.0

# Beta-trace y-limits, fixed per dataset (reference and zoom figures share
# them); both include 0 dB so the zero line is always visible.
TRACE_YLIM = {"stepup": (-4.5, 2.5), "jacobsen": (-3.5, 0.5)}
GPM_TRACE_YLIM = (-0.75, 0.75)

# Specification-curve y-limits, fixed per dataset (cover every
# universe's 95% CI and 0 dB).
SPEC_YLIM = {"stepup": (-0.5, 3.5), "jacobsen": (-0.25, 1.25)}

# --- Labels ---------------------------------------------------------------
# Dataset names, decision-node labels and node order come from src.config
# (one label map for figures, QC tables and docs).
DATASET_NAME = {k: v.capitalize() for k, v in DATASET_LABELS.items()}
ROI_LABEL = "linear ROI centred on Cz"
NORM_LABEL = "dB re standing baseline"
ERSP_CBAR_LABEL = "ERSP (dB re standing baseline)"
GPM_CBAR_LABEL = "GPM (dB)"
SPEC_YLABEL = "Subject beta contrast\n(double-stance - swing, dB)"

# Row order of the decision nodes, as in Table 2.1 of the manuscript.
NODE_ORDER = MULTIVERSE_GRID


def option_label(decision: str, value: object) -> str:
    """Return the display label of one decision-node option.

    Parameters
    ----------
    decision : str
        Decision-node key in ``OPTION_LABELS``.
    value : object
        Option value; falls back to ``str(value)`` if it has no label.

    Returns
    -------
    str
        Display label.
    """
    return OPTION_LABELS[decision].get(value, str(value))


def universe_label(decisions: dict) -> str:
    """Return a one-line universe label, e.g. 'HP 0.5 Hz, ASR SD 20, IC liberal'.

    Uses the same label map as the specification curves.

    Parameters
    ----------
    decisions : dict
        Mapping with ``highpass_hz``, ``asr_mode`` and ``iclabel_rule``.

    Returns
    -------
    str
        Universe label.
    """
    return (
        f"HP {option_label('highpass_hz', float(decisions['highpass_hz']))} Hz, "
        f"ASR {option_label('asr_mode', decisions['asr_mode'])}, "
        f"IC {option_label('iclabel_rule', decisions['iclabel_rule'])}"
    )


def figure_title(dataset: str, content: str, n: int) -> str:
    """Return the standard figure title.

    Parameters
    ----------
    dataset : str
        Dataset key (``"stepup"`` or ``"jacobsen"``).
    content : str
        What the figure shows.
    n : int
        Number of subjects.

    Returns
    -------
    str
        Title text.
    """
    return f"{DATASET_NAME[dataset]} walking: {content} (n = {n})"


def figure_subtitle(fmin: float, fmax: float, norm: str = NORM_LABEL) -> str:
    """Return the standard figure subtitle (ROI, normalisation, band).

    Parameters
    ----------
    fmin, fmax : float
        Frequency range in Hz.
    norm : str
        Normalisation label.

    Returns
    -------
    str
        Subtitle text.
    """
    return f"{ROI_LABEL[0].upper() + ROI_LABEL[1:]}, {norm}, {fmin:g}-{fmax:g} Hz"


def scale_note(vlim: float, zero: str = "0 dB") -> str:
    """Return the colour-scale note text.

    Parameters
    ----------
    vlim : float
        Symmetric colour limit in dB.
    zero : str
        What white represents.

    Returns
    -------
    str
        Note text.
    """
    return f"Colour scale +/-{vlim:g} dB, white = {zero}"


def format_p(p: float) -> str:
    """Return a p value as figure text: "p < 0.001" below 0.001, else three significant digits.

    Parameters
    ----------
    p : float
        p value.

    Returns
    -------
    str
        Text such as "p = 0.0123" or "p < 0.001".
    """
    return "p < 0.001" if p < 0.001 else f"p = {p:.3g}"


def heatmap_topo_note(vlim: float, topo_vlim: float) -> str:
    """Return the scale note of a figure whose topographies have their own range.

    Parameters
    ----------
    vlim : float
        Heatmap colour limit in dB.
    topo_vlim : float
        Topography colour limit in dB.

    Returns
    -------
    str
        Note text.
    """
    return f"Heatmap scale +/-{vlim:g} dB, Topography scale +/-{topo_vlim:g} dB, white = 0 dB"


def add_header(
    fig: plt.Figure,
    title: str,
    subtitle: str | None = None,
    note: str | None = None,
    top: float = 0.99,
) -> None:
    """Add a bold title, optional subtitle and small note at the figure top.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    title : str
        Bold title.
    subtitle : str | None
        Optional subtitle below the title.
    note : str | None
        Optional small italic note below the subtitle.
    top : float
        Figure y-coordinate of the title.
    """
    y = top
    fig.text(0.5, y, title, ha="center", va="top", fontsize=FS_TITLE, fontweight="bold")
    y -= 0.03
    if subtitle:
        fig.text(0.5, y, subtitle, ha="center", va="top", fontsize=FS_LABEL)
        y -= 0.027
    if note:
        fig.text(0.5, y, note, ha="center", va="top", fontsize=FS_NOTE, style="italic")


def apply_rc() -> None:
    """Apply the shared font sizes and dpi to matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "font.size": FS_TICK,
            "axes.titlesize": FS_LABEL,
            "axes.labelsize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_NOTE,
            "savefig.dpi": DPI,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    """Save a figure at the shared dpi, close it and print the path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : Path
        Output file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# --- Colour scale ---------------------------------------------------------
def ersp_norm(vlim: float) -> Normalize:
    """Return a symmetric colour normalisation.

    Parameters
    ----------
    vlim : float
        Symmetric colour limit.

    Returns
    -------
    matplotlib.colors.Normalize
        Normalisation from ``-vlim`` to ``vlim``.
    """
    return Normalize(vmin=-vlim, vmax=vlim)


def clip_fraction(values: Iterable[np.ndarray], vlim: float) -> float:
    """Return the fraction of values whose magnitude exceeds ``vlim``.

    Parameters
    ----------
    values : iterable of np.ndarray
        Arrays pooled together.
    vlim : float
        Symmetric colour limit.

    Returns
    -------
    float
        Clipped fraction (0-1).
    """
    v = np.concatenate([np.ravel(a) for a in values])
    return float((np.abs(v) > vlim).mean())


def rule_vlim(
    values: Iterable[np.ndarray],
    candidates: Sequence[float] = VLIM_CANDIDATES,
    max_clip: float = MAX_CLIP,
) -> float | None:
    """Return the smallest candidate limit leaving at most ``max_clip`` clipped.

    Parameters
    ----------
    values : iterable of np.ndarray
        Arrays pooled together.
    candidates : sequence of float
        Candidate limits, in increasing order.
    max_clip : float
        Maximum allowed clipped fraction.

    Returns
    -------
    float | None
        First candidate that satisfies the rule, or None if none does.
    """
    return next((c for c in candidates if clip_fraction(values, c) <= max_clip), None)


def report_clipping(
    name: str,
    values: Iterable[np.ndarray],
    vlim: float,
    candidates: Sequence[float] = VLIM_CANDIDATES,
) -> float:
    """Print the fixed range, % clipped and what the rule would pick today.

    Parameters
    ----------
    name : str
        Label used in the printed line.
    values : iterable of np.ndarray
        Arrays pooled together.
    vlim : float
        Fixed colour limit in use.
    candidates : sequence of float
        Candidate limits passed to `rule_vlim`.

    Returns
    -------
    float
        Clipped fraction at ``vlim``.
    """
    frac = clip_fraction(values, vlim)
    rule = rule_vlim(values, candidates)
    if rule is None:
        flag = f"  [NOTE: no candidate clips <= {100 * MAX_CLIP:g} %]"
    else:
        flag = "" if rule == vlim else f"  [NOTE: rule now gives +/-{rule}]"
    print(f"  Colour range {name}: +/-{vlim:g} dB, {100 * frac:.2f}% clipped{flag}")
    return frac


def add_ersp_colorbar(
    fig: plt.Figure,
    mappable: ScalarMappable,
    axes: plt.Axes | Sequence[plt.Axes] | None,
    vlim: float,
    label: str = ERSP_CBAR_LABEL,
    high: str = "above baseline (ERS)",
    low: str = "below baseline (ERD)",
    **kw: object,
) -> Colorbar:
    """Add a colourbar with true dB ticks and small end labels.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    mappable : matplotlib.cm.ScalarMappable
        Image the colourbar describes.
    axes : Axes | sequence of Axes | None
        Axes to take space from; None uses the ``fig.colorbar`` default.
    vlim : float
        Symmetric colour limit in dB (sets the tick step).
    label : str
        Colourbar label.
    high, low : str
        Labels drawn above and below the bar.
    **kw
        Passed on to ``fig.colorbar``.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colourbar.
    """
    cbar = (
        fig.colorbar(mappable, ax=axes, **kw) if axes is not None else fig.colorbar(mappable, **kw)
    )
    step = 0.5 if vlim <= 1.5 else 1.0
    ticks = np.arange(-np.floor(vlim / step) * step, vlim + 1e-9, step)
    cbar.set_ticks(ticks)
    cbar.set_label(label, fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cbar.ax.text(
        0.5, 1.02, high, transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=FS_NOTE
    )
    cbar.ax.text(
        0.5, -0.02, low, transform=cbar.ax.transAxes, ha="center", va="top", fontsize=FS_NOTE
    )
    return cbar


# --- Gait-cycle panels ----------------------------------------------------
GAIT_XTICKS = [0, 20, 40, 60, 80, 100]
DS_COLOR = "#C4BFBF"
SWING_COLOR = "#D2D899"
TRACE_COLOR = "#2c5f8a"
# Black lines with a white halo stay visible on dark and light cells.
_HALO = [pe.withStroke(linewidth=2.6, foreground="white")]


def gait_events(anchors: Sequence[float]) -> list[tuple[float, str]]:
    """Return gait-event positions (% cycle) and labels, RHS to RHS.

    Parameters
    ----------
    anchors : sequence of float
        ``(A_lto, A_lhs, A_rto)`` in % gait cycle.

    Returns
    -------
    list of (float, str)
        Event position and label.
    """
    a_lto, a_lhs, a_rto = anchors
    return [(0, "RHS"), (a_lto, "LTO"), (a_lhs, "LHS"), (a_rto, "RTO"), (100, "RHS")]


def phase_bars(anchors: Sequence[float]) -> list[tuple[float, float, str, str]]:
    """Return the gait-phase bar segments.

    Parameters
    ----------
    anchors : sequence of float
        ``(A_lto, A_lhs, A_rto)`` in % gait cycle.

    Returns
    -------
    list of (float, float, str, str)
        Left edge, width, colour and label of each phase.
    """
    a_lto, a_lhs, a_rto = anchors
    return [
        (0, a_lto, DS_COLOR, "DS"),
        (a_lto, a_lhs - a_lto, SWING_COLOR, "LLS"),
        (a_lhs, a_rto - a_lhs, DS_COLOR, "DS"),
        (a_rto, 100 - a_rto, SWING_COLOR, "RLS"),
    ]


def freq_ticks(freqs: Sequence[float]) -> list[int]:
    """Return the standard frequency ticks that fall inside ``freqs``.

    Parameters
    ----------
    freqs : sequence of float
        Frequency axis in Hz (ascending).

    Returns
    -------
    list of int
        Tick positions in Hz.
    """
    return [f for f in (8, 13, 20, 30, 40, 50, 60) if freqs[0] <= f <= freqs[-1]]


def draw_heatmap(
    ax: plt.Axes,
    arr: np.ndarray,
    freqs: Sequence[float],
    vlim: float,
    anchors: Sequence[float],
    beta_band: Sequence[float] = (13, 30),
    ylabel: bool = True,
) -> AxesImage:
    """Draw a frequency x gait-cycle heatmap with event and beta-band lines.

    x-extent is half-bin corrected: the n columns sit at their true
    gait-cycle positions 0..100, matching the event lines exactly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    arr : np.ndarray
        ERSP map, shape (n_freqs, n_points).
    freqs : sequence of float
        Frequency axis in Hz.
    vlim : float
        Symmetric colour limit in dB.
    anchors : sequence of float
        ``(A_lto, A_lhs, A_rto)`` in % gait cycle.
    beta_band : sequence of float
        Frequencies (Hz) at which horizontal lines are drawn.
    ylabel : bool
        Whether to label the y-axis.

    Returns
    -------
    matplotlib.image.AxesImage
        The heatmap image.
    """
    dx = 100.0 / (arr.shape[-1] - 1)
    im = ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        cmap=CMAP,
        norm=ersp_norm(vlim),
        extent=[-dx / 2, 100 + dx / 2, freqs[0] - 0.5, freqs[-1] + 0.5],
        zorder=1,
    )
    for pct, _ in gait_events(anchors):
        ax.axvline(pct, color="black", linewidth=1.0, linestyle=":", zorder=3, path_effects=_HALO)
    for f in beta_band:
        ax.axhline(f, color="black", linewidth=0.9, linestyle="--", zorder=4, path_effects=_HALO)
    ax.set_xlim(0, 100)
    ax.set_ylim(freqs[0] - 0.5, freqs[-1] + 0.5)
    ax.set_yticks(freq_ticks(freqs))
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.spines["bottom"].set_visible(False)
    if ylabel:
        ax.set_ylabel("Frequency (Hz)", fontsize=FS_LABEL)
    return im


def draw_beta_trace(
    ax: plt.Axes,
    trace: np.ndarray,
    ylim: Sequence[float],
    anchors: Sequence[float],
    ylabel: str = "Beta 13-30 Hz\n(dB)",
) -> None:
    """Draw a beta-power trace over the gait cycle with event lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    trace : np.ndarray
        Beta power (dB) over the gait cycle.
    ylim : sequence of float
        y-axis limits.
    anchors : sequence of float
        ``(A_lto, A_lhs, A_rto)`` in % gait cycle.
    ylabel : str
        y-axis label; empty for none.
    """
    x = np.linspace(0, 100, trace.shape[-1])
    ax.plot(x, trace, color=TRACE_COLOR, linewidth=1.6, zorder=2)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
    for pct, _ in gait_events(anchors):
        ax.axvline(pct, color="black", linewidth=0.8, linestyle=":", zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_event_row(ax: plt.Axes, anchors: Sequence[float]) -> None:
    """Draw the row of gait-event ticks and labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    anchors : sequence of float
        ``(A_lto, A_lhs, A_rto)`` in % gait cycle.
    """
    ax.set_ylim(-0.3, 1.0)
    ax.plot([0, 100], [0.5, 0.5], color="black", linewidth=1.0, clip_on=False)
    for pct, label in gait_events(anchors):
        ax.plot([pct, pct], [0.2, 0.8], color="black", linewidth=1.0)
        ax.text(pct, 0.15, label, ha="center", va="top", fontsize=FS_NOTE)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_phase_bar(ax: plt.Axes, anchors: Sequence[float]) -> None:
    """Draw the gait-phase bar (DS / LLS / DS / RLS) and the x-axis label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    anchors : sequence of float
        ``(A_lto, A_lhs, A_rto)`` in % gait cycle.
    """
    ax.set_ylim(0, 1)
    for left, width, color, label in phase_bars(anchors):
        ax.barh(0.5, width, left=left, height=1.0, color=color, align="center", edgecolor="none")
        ax.text(left + width / 2, 0.5, label, ha="center", va="center", fontsize=FS_NOTE)
    ax.set_xlabel("Gait cycle (%)", fontsize=FS_LABEL, labelpad=4)
    ax.set_yticks([])
    ax.set_xticks(GAIT_XTICKS)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for spine in ax.spines.values():
        spine.set_visible(False)


def gait_column(
    fig: plt.Figure,
    subplot_spec: SubplotSpec,
    with_trace: bool = True,
    hspace: float = 0.12,
) -> list[plt.Axes]:
    """Create the axes for one gait-cycle column.

    Rows: heatmap, optional beta trace, event row, phase bar; all share the
    heatmap's x-axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    subplot_spec : matplotlib.gridspec.SubplotSpec
        Grid cell to subdivide.
    with_trace : bool
        Whether to include the beta-trace row.
    hspace : float
        Vertical spacing between rows.

    Returns
    -------
    list of matplotlib.axes.Axes
        Axes from top to bottom.
    """
    import matplotlib.gridspec as gridspec

    ratios = [10, 3, 1, 1] if with_trace else [10, 1, 1]
    gs = gridspec.GridSpecFromSubplotSpec(
        len(ratios), 1, subplot_spec=subplot_spec, height_ratios=ratios, hspace=hspace
    )
    ax_heat = fig.add_subplot(gs[0])
    axes = [ax_heat] + [fig.add_subplot(gs[i], sharex=ax_heat) for i in range(1, len(ratios))]
    return axes
