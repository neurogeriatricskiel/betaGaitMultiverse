"""Gait events per subject for reference stage 1 (prepana01): load the events."""

from pathlib import Path
from typing import NamedTuple

import numpy as np

from src import config
from src.gait_cycles import (
    detect_gait_events,
    heel_relative_signal,
    load_gait_events_from_tsv,
    load_motion,
)


class GaitEvents(NamedTuple):
    """Event sample indices of one subject and the traces they were detected on."""

    lhs: np.ndarray
    lto: np.ndarray
    rhs: np.ndarray
    rto: np.ndarray
    sfreq: float
    left_trace: np.ndarray | None  # filtered heel-minus-pelvis signal (motion capture only)
    right_trace: np.ndarray | None


def events_from_motion_capture(subject: str, raw_dir: Path) -> GaitEvents | None:
    """Detect heel strikes and toe-offs from the treadmill motion-capture recording.

    Parameters
    ----------
    subject : str
        Subject ID without the "sub-" prefix.
    raw_dir : pathlib.Path
        BIDS root of the raw data.

    Returns
    -------
    GaitEvents or None
        None if the motion file is missing.
    """
    motion_file = raw_dir / f"sub-{subject}" / "motion" / f"sub-{subject}_task-CS.tsv"
    if not motion_file.exists():
        print(f"  Motion file not found, skipping: {motion_file}")
        return None
    df, marker_cols = load_motion(motion_file)  # column names resolved from channels.tsv
    print(f"  Loaded {len(df)} samples x {len(df.columns)} channels")
    pelvis = df[marker_cols["PELV"]].values
    left = heel_relative_signal(df[marker_cols["LHEE"]].values, pelvis)
    right = heel_relative_signal(df[marker_cols["RHEE"]].values, pelvis)
    lhs, lto, left_trace = detect_gait_events(left, config.MOCAP_SFREQ)
    rhs, rto, right_trace = detect_gait_events(right, config.MOCAP_SFREQ)
    return GaitEvents(lhs, lto, rhs, rto, config.MOCAP_SFREQ, left_trace, right_trace)


def events_from_bids_events(subject: str, raw_dir: Path) -> GaitEvents | None:
    """Read the overground dataset's own gait events from the BIDS events.tsv.

    Parameters
    ----------
    subject : str
        Subject ID without the "sub-" prefix.
    raw_dir : pathlib.Path
        BIDS root of the raw data.

    Returns
    -------
    GaitEvents or None
        Events within the steady walking segment; None if events.tsv is missing.
    """
    events_tsv = (
        raw_dir / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{config.TASK_NAME}_events.tsv"
    )
    if not events_tsv.exists():
        print(f"  events.tsv not found, skipping: {events_tsv}")
        return None
    lhs, lto, rhs, rto, seg_t0, seg_t1 = load_gait_events_from_tsv(
        events_tsv,
        config.SEGMENT_START_VALUE,
        config.SEGMENT_END_VALUE,
        config.GAIT_EVENT_VALUES,
    )
    print(
        f"  Segment [{seg_t0:.1f}, {seg_t1:.1f}] s  "
        f"events: LHS={len(lhs)} LTO={len(lto)} RHS={len(rhs)} RTO={len(rto)}"
    )
    return GaitEvents(lhs, lto, rhs, rto, config.EEG_SFREQ_NATIVE, None, None)


def load_gait_events(subject: str, raw_dir: Path) -> GaitEvents | None:
    """Load one subject's gait events from the dataset's EVENT_SOURCE.

    Parameters
    ----------
    subject : str
        Subject ID without the "sub-" prefix.
    raw_dir : pathlib.Path
        BIDS root of the raw data.

    Returns
    -------
    GaitEvents or None
        None if the input file is missing.
    """
    if config.EVENT_SOURCE == "mocap":
        return events_from_motion_capture(subject, raw_dir)
    if config.EVENT_SOURCE == "events_tsv":
        return events_from_bids_events(subject, raw_dir)
    raise ValueError(f"Unknown EVENT_SOURCE: {config.EVENT_SOURCE!r}")
