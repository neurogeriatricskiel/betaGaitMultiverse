"""Reference group ERSP figure (prepana06): heatmap, beta trace, events and topographies."""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from src import plot_style as ps
from src.config import PIPELINE_TFR_FMAX, ROI_CENTER_CH, TFR_FMIN, TFR_N_POINTS
from src.ersp import BETA_FMAX, BETA_FMIN, phase_split_indices
from src.spatial_filter import TOPOMAP_SPHERE, apply_linear_roi, linear_roi_weights

FREQS = np.arange(TFR_FMIN, int(PIPELINE_TFR_FMAX) + 1)
BETA_MASK = (FREQS >= BETA_FMIN) & (FREQS <= BETA_FMAX)
TOPO_KEYS = ("beta_whole", "beta_swing", "beta_ds")
TOPO_TITLES = ["Beta 13-30 Hz, whole cycle", "Beta 13-30 Hz, swing", "Beta 13-30 Hz, double stance"]


def print_event_positions(subject: str, ersp_roi: np.ndarray, cycles: pd.DataFrame) -> None:
    """Print the subject's ROI mean and own mean event positions (the figure uses the anchors)."""
    dur = cycles["rhs_end_s"].values - cycles["rhs_start_s"].values
    pct = {
        ev: float(np.mean((cycles[f"{ev}_s"].values - cycles["rhs_start_s"].values) / dur * 100))
        for ev in ("lto", "lhs", "rto")
    }
    print(
        f"  sub-{subject}: Linear ROI mean={ersp_roi.mean():+.2f} dB  "
        f"LTO={pct['lto']:.1f}%  LHS={pct['lhs']:.1f}%  RTO={pct['rto']:.1f}%"
    )


def load_group(cohort: list[str], dirs: dict, anchors: tuple[float, float, float]) -> dict:
    """Per-subject ROI ERSP, beta topographies and ROI phase values of the cohort.

    Parameters
    ----------
    cohort : list of str
        Subjects to load (missing files are skipped).
    dirs : dict
        Dataset folders.
    anchors : tuple of float
        Anchors the ERSP arrays were warped to.

    Returns
    -------
    dict
        Lists "ersp", "beta_whole", "beta_swing", "beta_ds", "roi_ds", "roi_sw"
        and the channel "info" of the last subject loaded.
    """
    double_stance_idx, swing_idx = phase_split_indices(anchors, n_points=TFR_N_POINTS)
    g = {k: [] for k in ("ersp", *TOPO_KEYS, "roi_ds", "roi_sw")}
    g["info"] = None
    for subject in cohort:
        try:
            ersp = np.load(dirs["ersp"] / f"sub-{subject}_ersp_beta.npy")  # (n_ch, n_freqs, 101)
            cycles = pd.read_csv(dirs["gaitepochs"] / f"sub-{subject}_cycles_kept.tsv", sep="\t")
            raw_path = dirs["clean"] / f"sub-{subject}_desc-icaClean_concat_raw.fif"
            raw_ref = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
            continue
        weights_path = dirs["ersp"] / f"sub-{subject}_roi_weights.npy"
        if weights_path.exists():
            weights = np.load(weights_path)
        else:
            weights = linear_roi_weights(raw_ref.info, center_ch=ROI_CENTER_CH)
        ersp_roi = apply_linear_roi(ersp, weights)  # (n_freqs, 101)
        ersp_beta = ersp[:, BETA_MASK, :]  # per-channel beta topographies, prepana05 windows
        g["beta_whole"].append(ersp_beta.mean(axis=(1, 2)))
        g["beta_swing"].append(ersp_beta[:, :, swing_idx].mean(axis=(1, 2)))
        g["beta_ds"].append(ersp_beta[:, :, double_stance_idx].mean(axis=(1, 2)))
        g["roi_ds"].append(float(ersp_roi[BETA_MASK][:, double_stance_idx].mean()))
        g["roi_sw"].append(float(ersp_roi[BETA_MASK][:, swing_idx].mean()))
        g["ersp"].append(ersp_roi)
        g["info"] = raw_ref.info
        print_event_positions(subject, ersp_roi, cycles)
    return g


def group_topographies(g: dict) -> list[np.ndarray]:
    """Group beta topographies (whole cycle, swing, double stance).

    Only subjects whose channel count matches the first subject are averaged.
    """
    n_ch_ref = g["beta_whole"][0].shape[0]
    return [
        np.mean(np.stack([t for t in g[key] if t.shape[0] == n_ch_ref]), axis=0)
        for key in TOPO_KEYS
    ]


def draw_topographies(
    fig: Figure, subplot_spec: SubplotSpec, topos: list[np.ndarray], info_ref: mne.Info, vlim: float
) -> object:
    """Three beta topographies stacked in one column; returns the last image (for a colour bar)."""
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=subplot_spec, hspace=0.35)
    for i, (data, title) in enumerate(zip(topos, TOPO_TITLES, strict=True)):
        ax = fig.add_subplot(gs_right[i])
        im, _ = mne.viz.plot_topomap(
            data,
            info_ref,
            axes=ax,
            show=False,
            cmap=ps.CMAP,
            vlim=(-vlim, vlim),
            contours=4,
            sphere=TOPOMAP_SPHERE,
        )
        ax.set_title(title, fontsize=ps.FS_LABEL)
    return im


def render_reference_figure(
    ersp_group: np.ndarray,
    topos: list[np.ndarray],
    info_ref: mne.Info,
    anchors: tuple[float, float, float],
    n_subjects: int,
    dataset: str,
    out_path: Path,
) -> None:
    """Save heatmap + beta trace + events + phase bar, three topographies and two colour bars.

    Parameters
    ----------
    ersp_group : numpy.ndarray, shape (n_freqs, 101)
        Group ROI ERSP, dB.
    topos : list of numpy.ndarray
        Output of ``group_topographies``.
    info_ref : mne.Info
        Channel positions for the topographies.
    anchors : tuple of float
        Event positions, % of the cycle.
    n_subjects : int
        Subjects averaged.
    dataset : str
        "stepup" or "jacobsen".
    out_path : pathlib.Path
        PNG path.
    """
    vlim = ps.ERSP_VLIM[(dataset, "reference")]
    topo_vlim = ps.TOPO_VLIM[dataset]
    ps.report_clipping("reference heatmap", [ersp_group], vlim)
    ps.report_clipping("reference topographies (channels)", topos, topo_vlim)
    beta_trace = ersp_group[BETA_MASK].mean(axis=0)
    trace_ylim = ps.TRACE_YLIM[dataset]
    print(f"  Beta trace {beta_trace.min():+.3f} .. {beta_trace.max():+.3f} dB  (y {trace_ylim})")

    ps.apply_rc()
    fig = plt.figure(figsize=ps.FIGSIZE["reference"])
    gs_outer = gridspec.GridSpec(
        1,
        4,
        width_ratios=[2.6, 0.05, 1.0, 0.05],
        wspace=0.3,
        top=0.88,
        bottom=0.06,
        left=0.07,
        right=0.93,
        figure=fig,
    )
    ax_heat, ax_beta, ax_events, ax_phases = ps.gait_column(fig, gs_outer[0])
    cax = fig.add_subplot(gs_outer[1])
    cax_topo = fig.add_subplot(gs_outer[3])
    ps.add_header(
        fig,
        ps.figure_title(dataset, "group ERSP over the gait cycle", n_subjects),
        ps.figure_subtitle(FREQS[0], FREQS[-1]),
        ps.heatmap_topo_note(vlim, topo_vlim),
    )
    im = ps.draw_heatmap(
        ax_heat, ersp_group, FREQS, vlim, anchors, beta_band=(BETA_FMIN, BETA_FMAX)
    )
    ps.draw_beta_trace(ax_beta, beta_trace, trace_ylim, anchors)
    ps.draw_event_row(ax_events, anchors)
    ps.draw_phase_bar(ax_phases, anchors)
    im_topo = draw_topographies(fig, gs_outer[2], topos, info_ref, topo_vlim)
    ps.add_ersp_colorbar(fig, im, None, vlim, cax=cax)
    ps.add_ersp_colorbar(fig, im_topo, None, topo_vlim, label=ps.TOPO_CBAR_LABEL, cax=cax_topo)
    ps.save(fig, out_path)
