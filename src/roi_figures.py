"""Figures of the channel-level checks (qc_roi_channel_sensitivity.py, qc_heelstrike_locked.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import ttest_rel

from src import plot_style as ps
from src.ersp import BETA_FMAX, BETA_FMIN
from src.reference_ersp import FREQS
from src.spatial_filter import TOPOMAP_SPHERE

if TYPE_CHECKING:
    from pathlib import Path

    from src.roi_diagnostics import PhaseMaps

PROFILE_FREQS = (8, 10, 13, 16, 20, 25, 30, 35, 40, 50, 60)
CONTRAST_CBAR_LABEL = "Beta double stance - swing (dB)"


def profile_lines(maps: PhaseMaps, channel: str) -> list[str]:
    """Report lines of the double-stance, swing and difference profile at one channel."""
    i = maps.ch_names.index(channel)
    ds, sw, diff = (
        maps.ds[:, i].mean(0),
        maps.sw[:, i].mean(0),
        (maps.ds[:, i] - maps.sw[:, i]).mean(0),
    )
    t = ttest_rel(maps.ds[:, i], maps.sw[:, i], axis=0)[0]
    lines = [
        f"  {channel}: DS-swing peak {diff.max():+.2f} dB at {FREQS[np.argmax(diff)]:.0f} Hz; "
        f"monotonic increase 8-60 Hz: {bool(np.all(np.diff(diff) >= 0))}",
        "     f(Hz):    " + " ".join(f"{f:>6.0f}" for f in PROFILE_FREQS),
    ]
    for label, arr in (("DS", ds), ("swing", sw), ("DS-sw", diff), ("t", t)):
        cols = " ".join(f"{arr[np.argmin(np.abs(FREQS - f))]:>+6.2f}" for f in PROFILE_FREQS)
        lines.append(f"     {label:<9} {cols}")
    return lines


def draw_freq_profile(maps: PhaseMaps, dataset: str, out_path: Path) -> list[str]:
    """QC plot of the frequency profile at Fpz and Cz; returns the report lines."""
    ps.apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    lines = []
    for ax, channel in zip(axes, ("Fpz", "Cz"), strict=True):
        i = maps.ch_names.index(channel)
        ax.plot(FREQS, maps.ds[:, i].mean(0), label="double stance")
        ax.plot(FREQS, maps.sw[:, i].mean(0), label="swing")
        ax.plot(FREQS, (maps.ds[:, i] - maps.sw[:, i]).mean(0), "k", label="DS - swing")
        ax.axvspan(BETA_FMIN, BETA_FMAX, color="0.9")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(channel)
        ax.set_xlabel("Frequency (Hz)")
        lines += profile_lines(maps, channel)
    axes[0].set_ylabel("ERSP (dB re standing baseline)")
    axes[0].legend(fontsize=ps.FS_NOTE)
    fig.suptitle(ps.figure_title(dataset, "ERSP by frequency at Fpz and Cz", len(maps.subjects)))
    ps.save(fig, out_path)
    return lines


def draw_contrast_topography(
    contrast: np.ndarray, info: mne.Info, dataset: str, n: int, out_path: Path
) -> None:
    """Supplementary topography of the per-channel beta contrast, unclipped.

    Parameters
    ----------
    contrast : numpy.ndarray, shape (n_channels,)
        Group-mean 13-30 Hz double stance minus swing per channel, dB.
    info : mne.Info
        Channel positions.
    dataset : str
        "stepup" or "jacobsen".
    n : int
        Subjects averaged.
    out_path : pathlib.Path
        PNG path.

    Notes
    -----
    The colour limit is the largest absolute channel value (no clipping);
    ``extrapolate="local"`` keeps the map within the electrode hull.
    """
    vlim = float(np.abs(contrast).max())
    ps.apply_rc()
    fig = plt.figure(figsize=ps.FIGSIZE["roi_topo"])
    ax = fig.add_axes([0.05, 0.08, 0.72, 0.72])
    cax = fig.add_axes([0.82, 0.18, 0.04, 0.52])
    ps.add_header(
        fig,
        ps.figure_title(dataset, "beta contrast per channel", n),
        f"Double stance - swing, {BETA_FMIN:.0f}-{BETA_FMAX:.0f} Hz, dB re standing baseline",
        f"Colour scale +/-{vlim:.2f} dB (largest channel value, unclipped), white = 0 dB",
    )
    im, _ = mne.viz.plot_topomap(
        contrast, info, axes=ax, show=False, cmap=ps.CMAP, vlim=(-vlim, vlim), contours=4,
        sphere=TOPOMAP_SPHERE, extrapolate="local",
    )  # fmt: skip
    ps.add_ersp_colorbar(
        fig, im, None, vlim, label=CONTRAST_CBAR_LABEL, high="double stance > swing",
        low="swing > double stance", cax=cax,
    )  # fmt: skip
    ps.save(fig, out_path)


def draw_heelstrike(
    averages: dict, times_ms: np.ndarray, dataset: str, n: int, out_path: Path
) -> None:
    """QC plot of the heel-strike-locked group averages (one panel per input, Fpz and Cz)."""
    ps.apply_rc()
    fig, axes = plt.subplots(1, len(averages), figsize=(11, 4), sharey=True)
    for ax, (key, group) in zip(axes, averages.items(), strict=True):
        for row, channel in enumerate(("Fpz", "Cz")):
            ax.plot(times_ms, group[row], label=channel)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(key)
        ax.set_xlabel("Time from heel strike (ms)")
    axes[0].set_ylabel("Amplitude (uV), 1-40 Hz")
    axes[0].legend()
    fig.suptitle(ps.figure_title(dataset, "heel-strike-locked average", n))
    ps.save(fig, out_path)
