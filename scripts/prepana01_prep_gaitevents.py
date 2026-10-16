"""
prepana01_prep_gaitevents.py -- reference pipeline, stage 1: gait events -> gait cycles.

Purpose
    Build each subject's gait-event table and validated gait-cycle table.
    Treadmill (EVENT_SOURCE "mocap"): heel strikes / toe-offs detected from
    the heel-minus-pelvis motion-capture signal. Overground ("events_tsv"):
    the dataset's own accelerometer-derived events from the BIDS events.tsv,
    restricted to the steady walking segment. Both feed the same cycle
    validation (LTO < LHS < RTO, one of each per RHS-RHS window).
Inputs
    datasets/<ds>/d00_raw/sub-*/motion/*_task-CS.tsv        (treadmill)
    datasets/<ds>/d00_raw/sub-*/eeg/*_events.tsv            (overground)
Outputs
    datasets/<ds>/d01_gaitevents/sub-*_events.tsv, sub-*_cycles.tsv
    datasets/<ds>/d01_gaitevents/sub-*_gait_qc.png, sub-*_segment_10s.png (treadmill)
    datasets/<ds>/d00_qc/sub-*_gait_events_qc.json
Manuscript
    Methods 2.2 (gait events and cycles).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/prepana01_prep_gaitevents.py
"""

import traceback

import pandas as pd

from src import config
from src.config import DATASET, SUBJECTS
from src.gait_cycles import (
    build_events_dataframe,
    event_quality_report,
    extract_valid_gait_cycles,
    plot_gait_qc,
    plot_gait_segment,
)
from src.gait_events import GaitEvents, load_gait_events
from src.paths import get_dataset_dirs
from src.qc import gait_events_qc, log_qc
from src.resume import stage_already_done


def save_qc_plots(events: GaitEvents, cycles_df: pd.DataFrame, out_dir, subject: str) -> None:
    """Save the gait QC plot and a 10 s segment plot (needs a continuous kinematic trace)."""
    if events.left_trace is None:
        print("\n  QC plots skipped (no continuous kinematic trace for this event source).")
        return
    traces = (events.left_trace, events.right_trace, events.lhs, events.lto, events.rhs, events.rto)
    qc_out = out_dir / f"sub-{subject}_gait_qc.png"
    plot_gait_qc(*traces, cycles_df, config.MOCAP_SFREQ, qc_out)
    print(f"\n  Saved QC plot -> {qc_out.name}")
    segment_out = out_dir / f"sub-{subject}_segment_10s.png"
    plot_gait_segment(*traces, cycles_df, config.MOCAP_SFREQ, segment_out, t_start=10.0)
    print(f"  Saved segment -> {segment_out.name}")


def process_subject(subject: str, dirs: dict) -> None:
    """Write one subject's events and cycles tables, QC json and QC plots."""
    events_out = dirs["gait_events"] / f"sub-{subject}_events.tsv"
    cycles_out = dirs["gait_events"] / f"sub-{subject}_cycles.tsv"
    if stage_already_done(
        [events_out, cycles_out], validate=lambda: pd.read_csv(cycles_out, sep="\t")
    ):
        print(f"  Already complete -- skipping sub-{subject}")
        return
    events = load_gait_events(subject, dirs["raw"])
    if events is None:
        return
    ev = (events.lhs, events.lto, events.rhs, events.rto, events.sfreq)
    events_df = build_events_dataframe(*ev)
    events_df.to_csv(events_out, sep="\t", index=False)
    print(f"\n  Saved events  -> {events_out.name}")
    cycles_df = extract_valid_gait_cycles(*ev)
    cycles_df.to_csv(cycles_out, sep="\t", index=False)
    print(f"  Saved cycles  -> {cycles_out.name}")

    flag, metrics = gait_events_qc(cycles_df)
    log_qc(qc_dir=dirs["qc"], subject=subject, stage="gait_events", flag=flag, metrics=metrics)
    print(
        f"  QC gait_events: {flag}  n={metrics['n_cycles']}  "
        f"dur={metrics['mean_dur_s']:.3f}+/-{metrics['std_dur_s']:.3f}s  cv={metrics['cv_dur']:.3f}"
    )
    print("\n  QC summary")
    for key, value in event_quality_report(events_df, cycles_df).items():
        print(f"    {key}: {value}")
    save_qc_plots(events, cycles_df, dirs["gait_events"], subject)


def main() -> None:
    """Run stage 1 for every configured subject."""
    dirs = get_dataset_dirs(DATASET)
    for subject in SUBJECTS:
        print(f"\n{'=' * 60}\n  Subject: {subject}\n{'=' * 60}")
        try:
            process_subject(subject, dirs)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
        except Exception as e:
            print(f"\n  [ERROR] sub-{subject}: unexpected error -- {e}")
            traceback.print_exc()
    print(f"\nDone. Processed {len(SUBJECTS)} subject(s): {SUBJECTS}")


if __name__ == "__main__":
    main()
