"""Gait power modulation (GPM) of the overground reference ERSP (supplement and QC only).

GPM(f, t) = ERSP(f, t) - mean_t ERSP(f, t), per subject and channel, as in
the source study (Jacobsen et al.). It removes each frequency's mean over
the gait cycle and so leaves the double-stance - swing contrast unchanged.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.image import AxesImage

from src import plot_style as ps
from src.config import ROI_CENTER_CH
from src.ersp import BETA_FMAX, BETA_FMIN
from src.reference_ersp import FREQS
from src.spatial_filter import apply_linear_roi

BETA_MASK = (FREQS >= BETA_FMIN) & (FREQS <= BETA_FMAX)


def gait_power_modulation(ersp: np.ndarray) -> np.ndarray:
    """Subtract each (channel, frequency) row's mean over the cycle axis.

    Parameters
    ----------
    ersp : numpy.ndarray, shape (..., n_points)
        ERSP over the gait cycle, dB.

    Returns
    -------
    numpy.ndarray
        GPM, same shape.
    """
    return ersp - ersp.mean(axis=-1, keepdims=True)


def subject_gpm(subject: str, dirs: dict) -> tuple[np.ndarray, np.ndarray]:
    """GPM of one subject at the linear Cz ROI and at Cz.

    Parameters
    ----------
    subject : str
        Subject ID.
    dirs : dict
        Dataset folders (d05_ersp and d03_clean are read).

    Returns
    -------
    roi, cz : numpy.ndarray, shape (n_freqs, n_points)
    """
    ersp = np.load(dirs["ersp"] / f"sub-{subject}_ersp_beta.npy")  # (n_ch, n_freqs, 101)
    weights = np.load(dirs["ersp"] / f"sub-{subject}_roi_weights.npy")
    raw_path = dirs["clean"] / f"sub-{subject}_desc-icaClean_concat_raw.fif"
    ch_names = mne.io.read_raw_fif(raw_path, preload=False, verbose=False).ch_names
    if len(ch_names) != ersp.shape[0]:
        raise RuntimeError(f"sub-{subject}: {len(ch_names)} channels vs ERSP {ersp.shape}")
    gpm = gait_power_modulation(ersp)
    return apply_linear_roi(gpm, weights), gpm[ch_names.index(ROI_CENTER_CH)]


def draw_gpm_column(
    fig: Figure,
    subplot_spec: SubplotSpec,
    gpm_group: np.ndarray,
    panel_title: str,
    anchors: tuple[float, float, float],
    first: bool,
) -> AxesImage:
    """Draw one GPM panel (heatmap, beta trace, events, phase bar); return the image."""
    ax_heat, ax_beta, ax_events, ax_phases = ps.gait_column(fig, subplot_spec)
    im = ps.draw_heatmap(
        ax_heat,
        gpm_group,
        FREQS,
        ps.GPM_VLIM,
        anchors,
        beta_band=(BETA_FMIN, BETA_FMAX),
        ylabel=first,
    )
    ax_heat.set_title(panel_title, fontsize=ps.FS_LABEL)
    ps.draw_beta_trace(
        ax_beta,
        gpm_group[BETA_MASK].mean(axis=0),
        ps.GPM_TRACE_YLIM,
        anchors,
        ylabel="Beta 13-30 Hz\nGPM (dB)" if first else None,
    )
    ps.draw_event_row(ax_events, anchors)
    ps.draw_phase_bar(ax_phases, anchors)
    return im


def render_gpm_figure(
    panels: list[tuple[str, np.ndarray]],
    anchors: tuple[float, float, float],
    n_subjects: int,
    dataset: str,
    out_path: Path,
) -> None:
    """Save the two-panel GPM figure (ROI and Cz) with one shared colour range.

    Parameters
    ----------
    panels : list of (str, numpy.ndarray)
        (panel title, group GPM (n_freqs, n_points)).
    anchors : tuple of float
        Event positions, % of the cycle.
    n_subjects : int
        Subjects averaged.
    dataset : str
        Dataset key, for the title.
    out_path : pathlib.Path
        PNG path.
    """
    vlim = ps.GPM_VLIM
    ps.report_clipping(
        "GPM (ROI + Cz panels)", [g for _, g in panels], vlim, candidates=ps.GPM_VLIM_CANDIDATES
    )
    for panel_title, gpm_group in panels:
        trace = gpm_group[BETA_MASK].mean(axis=0)
        print(
            f"  {panel_title}: beta trace peak {trace.max():+.3f} dB at {trace.argmax()}%  "
            f"trough {trace.min():+.3f} dB at {trace.argmin()}%  (y {ps.GPM_TRACE_YLIM})"
        )
    ps.apply_rc()
    fig = plt.figure(figsize=ps.FIGSIZE["gpm"])
    gs_outer = gridspec.GridSpec(
        1,
        3,
        width_ratios=[1, 1, 0.04],
        wspace=0.2,
        top=0.86,
        bottom=0.06,
        left=0.06,
        right=0.95,
        figure=fig,
    )
    ps.add_header(
        fig,
        ps.figure_title(dataset, "gait power modulation (GPM)", n_subjects),
        f"GPM = ERSP (dB re standing baseline) minus each frequency's mean over the gait cycle, "
        f"{FREQS[0]:g}-{FREQS[-1]:g} Hz",
        ps.scale_note(vlim, zero="0 dB (cycle mean)"),
    )
    im = None
    for col, (panel_title, gpm_group) in enumerate(panels):
        im = draw_gpm_column(fig, gs_outer[col], gpm_group, panel_title, anchors, col == 0)
    ps.add_ersp_colorbar(
        fig,
        im,
        None,
        vlim,
        label=ps.GPM_CBAR_LABEL,
        high="above cycle mean",
        low="below cycle mean",
        cax=fig.add_subplot(gs_outer[2]),
    )
    ps.save(fig, out_path)
