"""
prepana04_clean2gaitcycles.py -- reference pipeline, stage 4: gait-cycle segments.

Purpose
    Cut the ICA-corrected walking (CS) EEG into RHS-to-RHS gait-cycle
    segments at the recording's own sampling rate and reject cycles with
    non-finite samples or peak amplitude above AMP_THRESH
    (src.gait_segments). Time normalisation happens after the TFR, on the
    power time series (Seeber et al. 2015).
Inputs
    datasets/<ds>/d01_gaitevents/sub-*_cycles.tsv
    datasets/<ds>/d03_clean/sub-*_desc-icaClean_concat_raw.fif
Outputs
    datasets/<ds>/d04_gaitepochs/sub-*_gait_segments.npy (object array of (n_ch, n_samp))
    datasets/<ds>/d04_gaitepochs/sub-*_gait_sfreq.npy, sub-*_cycles_kept.tsv
    datasets/<ds>/d00_qc/sub-*_gait_epochs_qc.json
Manuscript
    Methods 2.5 (gait-cycle epochs, amplitude rejection).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/prepana04_clean2gaitcycles.py [subject]
"""

import mne
import numpy as np
import pandas as pd

from src.config import DATASET
from src.cycle_tfr import annotation_segment
from src.gait_segments import extract_cycle_segments, save_cycle_segments
from src.paths import get_dataset_dirs
from src.qc import gait_epochs_flag, log_qc
from src.resume import stage_already_done
from src.stage_runner import run_for_subjects, subjects_from_command_line


def describe_inputs(raw: mne.io.BaseRaw, cycles_df: pd.DataFrame) -> None:
    """Print the walking data, NaN channels and the gait-cycle time range."""
    eeg_dur = raw.times[-1]
    print(f"  EEG sfreq    : {raw.info['sfreq']:.0f} Hz")
    print(f"  EEG duration : {eeg_dur:.2f} s  ({raw.n_times} samples)")
    print(f"  EEG channels : {len(raw.ch_names)}")
    nan_by_ch = np.mean(np.isnan(raw.get_data()), axis=1)
    bad_chs = [raw.ch_names[i] for i, f in enumerate(nan_by_ch) if f > 0.01]
    if bad_chs:
        print(f"  Channels with >1% NaN ({len(bad_chs)}): {bad_chs}")
    gait_t0, gait_t1 = cycles_df["rhs_start_s"].min(), cycles_df["rhs_end_s"].max()
    print(f"\n  Gait cycles loaded : {len(cycles_df)}")
    print(f"  Gait time range    : {gait_t0:.2f} - {gait_t1:.2f} s")
    print(f"  Mean cycle duration: {cycles_df['duration_s'].mean():.3f} s")
    if gait_t0 < 0 or gait_t1 > eeg_dur + 1.0:
        print(
            f"\n  WARNING: gait cycles [{gait_t0:.1f}, {gait_t1:.1f}] s extend beyond "
            f"EEG [0.0, {eeg_dur:.1f}] s -- check synchronisation."
        )


def log_epochs_qc(n_extracted: int, n_total: int, seg_lengths, dirs: dict, subject: str) -> None:
    """Write the stage-4 QC record (cycles kept and rejected)."""
    reject_frac = 1.0 - n_extracted / n_total if n_total > 0 else 1.0
    mean_seg_len = float(np.mean(seg_lengths))
    flag = gait_epochs_flag(n_extracted, reject_frac)
    metrics = {
        "n_extracted": n_extracted,
        "n_total": n_total,
        "reject_frac": round(reject_frac, 3),
        "mean_seg_len": round(mean_seg_len, 1),
    }
    log_qc(qc_dir=dirs["qc"], subject=subject, stage="gait_epochs", flag=flag, metrics=metrics)
    print(
        f"  QC gait_epochs: {flag}  extracted={n_extracted}/{n_total}  "
        f"reject_frac={reject_frac:.3f}  mean_seg_len={mean_seg_len:.1f} samples"
    )


def process_subject(subject: str, dirs: dict) -> None:
    """Extract, save and QC the gait-cycle segments of one subject."""
    raw_path = dirs["clean"] / f"sub-{subject}_desc-icaClean_concat_raw.fif"
    cycles_path = dirs["gait_events"] / f"sub-{subject}_cycles.tsv"
    paths = tuple(
        dirs["gaitepochs"] / f"sub-{subject}_{name}"
        for name in ("gait_segments.npy", "gait_sfreq.npy", "cycles_kept.tsv")
    )
    if stage_already_done(
        list(paths),
        inputs=[raw_path, cycles_path],
        validate=lambda: np.load(paths[0], allow_pickle=True),
    ):
        print(f"  Already complete -- skipping sub-{subject}")
        return
    if not raw_path.exists():
        print(f"  ICA-corrected EEG not found, skipping: {raw_path.name}")
        return
    raw = annotation_segment(mne.io.read_raw_fif(raw_path, preload=True, verbose=False), "CS")
    cycles_df = pd.read_csv(cycles_path, sep="\t")
    if len(cycles_df) == 0:
        print("  ERROR: cycles.tsv is empty -- skipping subject.")
        return
    describe_inputs(raw, cycles_df)
    result = extract_cycle_segments(raw, cycles_df)
    n_extracted = len(result.segments)
    print(f"\n  Extracted  : {n_extracted} / {len(cycles_df)} cycles")
    print(f"  Skipped OOB: {result.n_out_of_bounds}\n  Skipped bad: {result.n_bad}")
    for reason in result.rejection_log:
        print(f"    rejected: {reason}")
    if n_extracted == 0:
        print("  No segments extracted -- skipping subject.")
        return
    seg_lengths = [s.shape[-1] for s in result.segments]
    print(f"  Segment lengths: min={min(seg_lengths)}  max={max(seg_lengths)} samples")
    save_cycle_segments(result.segments, raw.info["sfreq"], result.kept_rows, paths)
    print("\n  Saved -> " + ", ".join(p.name for p in paths))
    log_epochs_qc(n_extracted, len(cycles_df), seg_lengths, dirs, subject)


def main() -> None:
    """Run stage 4 for one or all subjects."""
    subjects = subjects_from_command_line("Gait-cycle segmentation (reference stage 4).")
    dirs = get_dataset_dirs(DATASET)
    run_for_subjects(subjects, lambda subject: process_subject(subject, dirs))


if __name__ == "__main__":
    main()
