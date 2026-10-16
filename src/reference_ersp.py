"""ERSP of one subject for reference stage 5 (prepana05).

Per gait cycle: Morlet power, edge crop, amplitude rejection and the
4-segment warp to the reference anchors (src.cycle_tfr, src.ersp); ERSP =
mean over cycles of 10*log10(P_cycle / P_standing); double-stance and
swing averages on the warped grid.
"""

from collections.abc import Sequence
from typing import NamedTuple

import mne
import numpy as np
import pandas as pd

from src.config import (
    AMP_THRESH,
    PIPELINE_TFR_FMAX,
    ROI_CENTER_CH,
    TFR_EDGE_CROP,
    TFR_FMIN,
    TFR_N_CYCLES_DIVISOR,
    TFR_N_POINTS,
)
from src.cycle_tfr import cycle_event_indices, cycle_power, log_ratio_per_cycle, standing_segment
from src.ersp import compute_standing_baseline, phase_split_indices, warp_cycle_to_grid
from src.qc import BASELINE_POWER_RANGE_V2, ersp_flag, log_qc
from src.spatial_filter import apply_linear_roi, linear_roi_weights, plot_weight_topography

FREQS = np.arange(TFR_FMIN, int(PIPELINE_TFR_FMAX) + 1, dtype=float)  # 8-60 Hz
N_CYCLES = FREQS / TFR_N_CYCLES_DIVISOR


class WarpedCycles(NamedTuple):
    """Warped power of the accepted cycles and the rejection counts."""

    power: list[np.ndarray]  # each (n_channels, n_freqs, TFR_N_POINTS)
    n_rejected_amplitude: int
    n_rejected_event_order: int


def standing_baseline(raw: mne.io.BaseRaw, subject: str) -> tuple[np.ndarray, list[str]]:
    """Standing power per channel and frequency, and the standing channel names.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        ICA-corrected concatenated recording.
    subject : str
        Subject ID, for error messages.

    Returns
    -------
    baseline : numpy.ndarray, shape (n_channels, n_freqs)
        Mean Morlet power over the STAND segment (end trimmed).
    channels : list of str
        Channel names of the standing data.
    """
    raw_stand = standing_segment(raw, subject)
    baseline = compute_standing_baseline(
        raw_stand, FREQS, N_CYCLES, edge_crop=TFR_EDGE_CROP, amp_thresh=AMP_THRESH
    )
    return baseline, list(raw_stand.ch_names)


def warp_cycles(
    segments: Sequence[np.ndarray],
    channel_index: list[int],
    sfreq: float,
    cycles: pd.DataFrame,
    anchors: tuple[float, float, float],
) -> WarpedCycles:
    """Morlet power of every cycle, warped so its LTO / LHS / RTO land on the anchors.

    Parameters
    ----------
    segments : sequence of numpy.ndarray
        Gait-cycle segments (n_channels, n_samples), row-aligned with ``cycles``.
    channel_index : list of int
        Rows of each segment to use (the standing channels, in order).
    sfreq : float
        Sampling rate of the segments, Hz.
    cycles : pandas.DataFrame
        Kept cycles with rhs_start_s, lto_s, lhs_s, rto_s.
    anchors : tuple of float
        (LTO, LHS, RTO) target positions, % of the cycle.

    Returns
    -------
    WarpedCycles
    """
    warped, n_amplitude, n_event_order = [], 0, 0
    for i, segment in enumerate(segments):
        segment = segment[channel_index]
        if np.abs(segment).max() > AMP_THRESH:
            n_amplitude += 1
            continue
        power, crop = cycle_power(segment, sfreq, FREQS, N_CYCLES)
        lto, lhs, rto = cycle_event_indices(cycles.iloc[i], sfreq, crop)
        try:
            warped.append(
                warp_cycle_to_grid(power, lto, lhs, rto, anchors=anchors, n_points=TFR_N_POINTS)
            )
        except ValueError:
            n_event_order += 1
    return WarpedCycles(warped, n_amplitude, n_event_order)


def phase_averages(
    ersp_per_cycle: np.ndarray, anchors: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Double-stance and swing ERSP: mean over each phase's samples, then over cycles.

    Parameters
    ----------
    ersp_per_cycle : numpy.ndarray, shape (n_cycles, n_channels, n_freqs, n_points)
        Per-cycle ERSP on the warped grid.
    anchors : tuple of float
        The anchors the cycles were warped to.

    Returns
    -------
    double_stance, swing : numpy.ndarray, shape (n_channels, n_freqs)
    """
    double_stance_idx, swing_idx = phase_split_indices(anchors, n_points=TFR_N_POINTS)
    double_stance = ersp_per_cycle[:, :, :, double_stance_idx].mean(axis=-1).mean(axis=0)
    swing = ersp_per_cycle[:, :, :, swing_idx].mean(axis=-1).mean(axis=0)
    return double_stance, swing


def subject_ersp(
    raw: mne.io.BaseRaw,
    segments: Sequence[np.ndarray],
    sfreq: float,
    cycles: pd.DataFrame,
    anchors: tuple[float, float, float],
    subject: str,
) -> dict | None:
    """Full stage-5 computation for one subject.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        ICA-corrected concatenated recording (channel reference and baseline).
    segments : sequence of numpy.ndarray
        Gait-cycle segments from stage 4.
    sfreq : float
        Sampling rate of the segments, Hz.
    cycles : pandas.DataFrame
        Kept cycles, row-aligned with ``segments``.
    anchors : tuple of float
        Reference warp anchors, % of the cycle.
    subject : str
        Subject ID.

    Returns
    -------
    dict or None
        ersp (n_channels, n_freqs, n_points), double_stance, swing,
        baseline, n_cycles and rejection counts; None if no cycle survives.
    """
    baseline, stand_channels = standing_baseline(raw, subject)
    print(f"  Baseline shape: {baseline.shape}  range=[{baseline.min():.4e}, {baseline.max():.4e}]")
    channel_index = [raw.ch_names.index(ch) for ch in stand_channels if ch in raw.ch_names]
    cycles_out = warp_cycles(segments, channel_index, sfreq, cycles, anchors)
    print(f"  Gait cycles rejected by amplitude    : {cycles_out.n_rejected_amplitude}")
    print(f"  Gait cycles rejected by event order  : {cycles_out.n_rejected_event_order}")
    print(f"  Gait cycles accepted                 : {len(cycles_out.power)}")
    if not cycles_out.power:
        return None
    ersp_per_cycle = log_ratio_per_cycle(cycles_out.power, baseline)
    double_stance, swing = phase_averages(ersp_per_cycle, anchors)
    return {
        "ersp": ersp_per_cycle.mean(axis=0),
        "double_stance": double_stance,
        "swing": swing,
        "baseline": baseline,
        "stand_channels": stand_channels,
        "n_cycles": len(cycles_out.power),
    }


def save_roi_weights(info: mne.Info, dirs: dict, subject: str) -> np.ndarray | None:
    """Save the linear Cz-ROI weights and their topography; None on failure."""
    try:
        weights = linear_roi_weights(info, center_ch=ROI_CENTER_CH)
        np.save(dirs["ersp"] / f"sub-{subject}_roi_weights.npy", weights)
        topo_path = dirs["roi_topo"] / f"sub-{subject}_roi_weights_topo.png"
        plot_weight_topography(weights, info, subject, out_path=topo_path, center_ch=ROI_CENTER_CH)
        print(
            f"  ROI weights saved  max_ch={info.ch_names[weights.argmax()]}  "
            f"max_w={weights.max():.4f}"
        )
        return weights
    except Exception as e:
        print(f"  [WARN] ROI weights failed: {e}")
        return None


def print_phase_summary(result: dict, anchors: tuple[float, float, float]) -> None:
    """Print the phase windows and the sensorimotor (Cz, C3, C4) phase means."""
    a_lto, a_lhs, a_rto = anchors
    print(
        f"  Phase windows (% of gait cycle): DS1=[0,{a_lto:.1f}]  "
        f"SW1=[{a_lto:.1f},{a_lhs:.1f}]  DS2=[{a_lhs:.1f},{a_rto:.1f}]  SW2=[{a_rto:.1f},100]"
    )
    ds_idx, sw_idx = phase_split_indices(anchors, n_points=TFR_N_POINTS)
    channels = result["stand_channels"]
    smx = [channels.index(c) for c in ["Cz", "C3", "C4"] if c in channels]
    ersp = result["ersp"]
    print(f"  Double stance ERSP (sensorimotor mean): {ersp[smx][:, :, ds_idx].mean():+.2f} dB")
    print(f"  Swing         ERSP (sensorimotor mean): {ersp[smx][:, :, sw_idx].mean():+.2f} dB")


def log_ersp_qc(result: dict, weights: np.ndarray | None, dirs: dict, subject: str) -> None:
    """Write the stage-5 QC record (cycles, NaNs, range, ROI means, baseline check)."""
    ersp = result["ersp"]
    n_nan = int(np.isnan(ersp).sum())
    ersp_range = float(ersp.max() - ersp.min())
    baseline_mean = float(result["baseline"].mean())
    baseline_ok = BASELINE_POWER_RANGE_V2[0] < baseline_mean < BASELINE_POWER_RANGE_V2[1]
    if not baseline_ok:
        print(f"  [WARN] sub-{subject}: baseline mean {baseline_mean:.3e} outside plausible range")

    def roi_db(arr):
        return None if weights is None else round(float(apply_linear_roi(arr, weights).mean()), 2)

    roi_mean = float(apply_linear_roi(ersp, weights).mean()) if weights is not None else np.nan
    flag = ersp_flag(n_nan, result["n_cycles"], ersp_range, baseline_ok)
    metrics = {
        "n_cycles_used": result["n_cycles"],
        "n_nan": n_nan,
        "ersp_range_db": round(ersp_range, 2),
        "roi_mean_db": round(roi_mean, 2) if not np.isnan(roi_mean) else None,
        "double_stance_roi_db": roi_db(result["double_stance"]),
        "swing_roi_db": roi_db(result["swing"]),
        "baseline_mean_v2": float(round(baseline_mean, 20)),
        "baseline_ok": baseline_ok,
    }
    log_qc(qc_dir=dirs["qc"], subject=subject, stage="ersp", flag=flag, metrics=metrics)
    print(f"  QC ersp: {flag}  cycles={result['n_cycles']}  nan={n_nan}  range={ersp_range:.2f}dB")
