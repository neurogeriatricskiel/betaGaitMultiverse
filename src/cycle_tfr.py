"""Per-cycle time-frequency steps shared by the reference pipeline, the multiverse and QC.

One implementation of: cutting the STAND / CS segment of a concatenated
recording, Morlet power of one gait cycle with the edge crop, the cycle's
event positions in the cropped array, and the per-cycle log ratio to the
standing baseline. Used by prepana04/05, src.multiverse_pipeline (and so
mulana04) and src.averaging_order.
"""

import mne
import numpy as np
import pandas as pd

from src.config import BASELINE_EDGE_TRIM_S, TFR_EDGE_CROP


def annotation_segment(raw: mne.io.BaseRaw, description: str) -> mne.io.BaseRaw:
    """Return a copy of ``raw`` cropped to its single annotation ``description``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Concatenated recording with "STAND" and "CS" annotations.
    description : str
        Annotation to crop to.

    Returns
    -------
    mne.io.BaseRaw
        The annotated segment; the end is clipped to the recording.
    """
    annotations = [a for a in raw.annotations if a["description"] == description]
    if len(annotations) != 1:
        raise RuntimeError(f"Expected 1 {description} annotation, found {len(annotations)}")
    onset = annotations[0]["onset"]
    return raw.copy().crop(onset, min(onset + annotations[0]["duration"], raw.times[-1]))


def standing_segment(raw: mne.io.BaseRaw, subject: str) -> mne.io.BaseRaw:
    """Return the STAND segment without its last BASELINE_EDGE_TRIM_S seconds.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Concatenated recording.
    subject : str
        Subject ID, for the error message.

    Returns
    -------
    mne.io.BaseRaw
        Standing data used for the baseline.
    """
    raw_stand = annotation_segment(raw, "STAND")
    stand_tmax = raw_stand.times[-1] - BASELINE_EDGE_TRIM_S
    if stand_tmax <= 0:
        raise RuntimeError(
            f"sub-{subject}: standing segment too short after trimming "
            f"({raw_stand.times[-1]:.1f} s)"
        )
    return raw_stand.crop(tmax=stand_tmax)


def cycle_power(
    segment: np.ndarray, sfreq: float, freqs: np.ndarray, n_cycles: np.ndarray
) -> tuple[np.ndarray, int]:
    """Morlet power of one gait-cycle segment with TFR_EDGE_CROP removed at both edges.

    Parameters
    ----------
    segment : numpy.ndarray, shape (n_channels, n_samples)
        One RHS-to-RHS cycle.
    sfreq : float
        Sampling rate, Hz.
    freqs : numpy.ndarray
        Wavelet frequencies, Hz.
    n_cycles : numpy.ndarray
        Wavelet cycles per frequency.

    Returns
    -------
    power : numpy.ndarray, shape (n_channels, n_freqs, n_samples - 2 * crop)
        Cropped power.
    crop : int
        Samples removed at each edge.
    """
    power = mne.time_frequency.tfr_array_morlet(
        segment[np.newaxis],
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        zero_mean=True,
        verbose=False,
    )[0]
    crop = int(TFR_EDGE_CROP * power.shape[-1])
    return (power[..., crop:-crop] if crop else power), crop


def cycle_event_indices(cycle: pd.Series, sfreq: float, crop: int) -> tuple[int, int, int]:
    """LTO, LHS and RTO sample positions of one cycle within its edge-cropped power.

    Parameters
    ----------
    cycle : pandas.Series
        Cycle-table row with rhs_start_s, lto_s, lhs_s, rto_s (seconds).
    sfreq : float
        Sampling rate, Hz.
    crop : int
        Samples removed at each edge by ``cycle_power``.

    Returns
    -------
    tuple of int
        (lto, lhs, rto) indices.
    """
    start = int(round(cycle["rhs_start_s"] * sfreq))
    lto, lhs, rto = (
        int(round(cycle[k] * sfreq)) - start - crop for k in ("lto_s", "lhs_s", "rto_s")
    )
    return lto, lhs, rto


def log_ratio_per_cycle(warped_cycles: list[np.ndarray], baseline: np.ndarray) -> np.ndarray:
    """Per-cycle ERSP, 10*log10(P_cycle / P_standing), dB.

    Parameters
    ----------
    warped_cycles : list of numpy.ndarray, each (n_channels, n_freqs, n_points)
        Warped power of the accepted cycles.
    baseline : numpy.ndarray, shape (n_channels, n_freqs)
        Standing power.

    Returns
    -------
    numpy.ndarray, shape (n_cycles, n_channels, n_freqs, n_points)
    """
    stack = np.stack(warped_cycles)
    return 10 * np.log10(stack / baseline[np.newaxis, :, :, np.newaxis])
