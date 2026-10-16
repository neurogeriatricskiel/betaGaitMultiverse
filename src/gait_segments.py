"""Gait-cycle EEG segments for reference stage 4 (prepana04)."""

from pathlib import Path
from typing import NamedTuple

import mne
import numpy as np
import pandas as pd

from src.config import AMP_THRESH

MAX_LOGGED_REJECTIONS = 5


class CycleSegments(NamedTuple):
    """Kept gait-cycle segments of one subject and why the others were dropped."""

    segments: list[np.ndarray]  # each (n_channels, n_samples), RHS to RHS
    kept_rows: list[pd.Series]  # matching rows of the cycle table
    n_out_of_bounds: int
    n_bad: int  # non-finite or above AMP_THRESH
    rejection_log: list[str]  # first MAX_LOGGED_REJECTIONS reasons


def extract_cycle_segments(raw: mne.io.BaseRaw, cycles_df: pd.DataFrame) -> CycleSegments:
    """Cut each gait cycle (RHS to RHS) out of the walking data and reject bad ones.

    A cycle is dropped if it falls outside the recording, contains
    non-finite values, or exceeds AMP_THRESH on any channel.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Walking (CS) segment, preloaded.
    cycles_df : pandas.DataFrame
        Cycle table with rhs_start_s and rhs_end_s (seconds from walk onset).

    Returns
    -------
    CycleSegments
    """
    sfreq = raw.info["sfreq"]
    segments, kept_rows, log = [], [], []
    n_oob = n_bad = 0
    for idx, row in cycles_df.iterrows():
        t_start, t_end = float(row["rhs_start_s"]), float(row["rhs_end_s"])
        i_start, i_end = int(round(t_start * sfreq)), int(round(t_end * sfreq))
        if i_start < 0 or i_end > raw.n_times or i_end <= i_start:
            if len(log) < MAX_LOGGED_REJECTIONS:
                log.append(
                    f"OOB: cycle {idx} t=[{t_start:.3f},{t_end:.3f}]s "
                    f"-> samples [{i_start},{i_end}] (n_times={raw.n_times})"
                )
            n_oob += 1
            continue
        data = raw.get_data(start=i_start, stop=i_end)
        if not np.isfinite(data).all():
            if len(log) < MAX_LOGGED_REJECTIONS:
                log.append(f"NaN/Inf: cycle {idx} t=[{t_start:.3f},{t_end:.3f}]s")
            n_bad += 1
            continue
        peak_amp = np.max(np.abs(data))
        if peak_amp > AMP_THRESH:
            n_bad += 1
            if len(log) < MAX_LOGGED_REJECTIONS:
                log.append(f"Amplitude: cycle {idx} peak={peak_amp * 1e6:.1f} uV")
            continue
        segments.append(data)
        kept_rows.append(row)
    return CycleSegments(segments, kept_rows, n_oob, n_bad, log)


def save_cycle_segments(
    segments: list[np.ndarray],
    sfreq: float,
    kept_rows: list[pd.Series],
    paths: tuple[Path, Path, Path],
) -> None:
    """Save the segments (object array), their sampling rate and the kept cycle rows.

    Parameters
    ----------
    segments : list of numpy.ndarray
        Kept segments, each (n_channels, n_samples).
    sfreq : float
        Sampling rate, Hz.
    kept_rows : list of pandas.Series
        Cycle-table rows of the kept segments, in the same order.
    paths : tuple of pathlib.Path
        (segments .npy, sampling-rate .npy, kept-cycles .tsv).
    """
    out_seg, out_sfreq, out_meta = paths
    segments_arr = np.empty(len(segments), dtype=object)
    for i, seg in enumerate(segments):
        segments_arr[i] = seg
    np.save(out_seg, segments_arr, allow_pickle=True)
    np.save(out_sfreq, np.float64(sfreq))
    pd.DataFrame(kept_rows).reset_index(drop=True).to_csv(out_meta, sep="\t", index=False)
