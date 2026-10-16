"""
prepana05_gaitcycles2tfr.py -- reference pipeline, stage 5: gait cycles -> ERSP.

Purpose
    Per gait cycle: Morlet TFR (n_cycles = f / TFR_N_CYCLES_DIVISOR),
    TFR_EDGE_CROP edge crop, amplitude rejection, and a 4-segment
    piecewise-linear warp so LTO/LHS/RTO land on the reference anchors
    (src.ersp.load_reference_anchors: treadmill = frozen group medians,
    overground = fixed 18/50/68 %). ERSP = mean over cycles of
    10*log10(P_cycle / P_standing) (per-cycle log ratio before averaging;
    Makeig 1993, Delorme & Makeig 2004), standing power from the STAND
    segment minus its last BASELINE_EDGE_TRIM_S (src.reference_ersp).
    Also writes the double-stance and swing averages and the linear Cz
    ROI weights. At 13 Hz (n_cycles 6.5) the wavelet spans ~500 ms of a
    ~1 s cycle, so 0-25 % and 75-100 % are edge-affected at <= 16 Hz
    (Cohen 2014, ch. 13).
Inputs
    datasets/<ds>/d03_clean/sub-*_desc-icaClean_concat_raw.fif
    datasets/<ds>/d04_gaitepochs/sub-*_gait_segments.npy, _gait_sfreq.npy, _cycles_kept.tsv
    datasets/<ds>/d05_ersp/group_gait_event_anchors_frozen.json
Outputs
    datasets/<ds>/d05_ersp/sub-*_ersp_beta.npy (n_ch, n_freqs, TFR_N_POINTS),
    sub-*_ersp_double_stance.npy, sub-*_ersp_swing.npy (n_ch, n_freqs),
    sub-*_roi_weights.npy (n_ch,); d05_roi_topo/sub-*_roi_weights_topo.png;
    d00_qc/sub-*_ersp_qc.json
Manuscript
    Methods 2.5 (TFR, baseline, warp), 2.6 (ROI).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/prepana05_gaitcycles2tfr.py [subject]
    Set BETAGAIT_DATASETS_DIR to write to (and read from) another data folder.
"""

import mne
import numpy as np
import pandas as pd

from src.config import DATASET
from src.ersp import load_reference_anchors
from src.paths import get_dataset_dirs
from src.reference_ersp import log_ersp_qc, print_phase_summary, save_roi_weights, subject_ersp
from src.resume import stage_already_done
from src.stage_runner import run_for_subjects, subjects_from_command_line


def process_subject(subject: str, dirs: dict, anchors) -> None:
    """Compute, save and QC the ERSP of one subject."""
    clean_fif = dirs["clean"] / f"sub-{subject}_desc-icaClean_concat_raw.fif"
    seg_path = dirs["gaitepochs"] / f"sub-{subject}_gait_segments.npy"
    outputs = {k: dirs["ersp"] / f"sub-{subject}_ersp_{k}.npy" for k in ("double_stance", "swing")}
    out_beta = dirs["ersp"] / f"sub-{subject}_ersp_beta.npy"
    out_weights = dirs["ersp"] / f"sub-{subject}_roi_weights.npy"
    if stage_already_done(
        [*outputs.values(), out_beta, out_weights],
        inputs=[clean_fif, seg_path],
        validate=lambda: np.load(out_beta),
    ):
        print(f"  Already complete -- skipping sub-{subject}")
        return
    raw = mne.io.read_raw_fif(clean_fif, preload=True, verbose=False)
    segments = np.load(seg_path, allow_pickle=True)
    sfreq = float(np.load(dirs["gaitepochs"] / f"sub-{subject}_gait_sfreq.npy"))
    cycles = pd.read_csv(dirs["gaitepochs"] / f"sub-{subject}_cycles_kept.tsv", sep="\t")
    print(f"  Channels: {len(raw.ch_names)}  gait segments: {len(segments)}  sfreq={sfreq:.0f} Hz")
    if len(cycles) != len(segments):
        raise RuntimeError(f"cycles_kept.tsv has {len(cycles)} rows, segments {len(segments)}")
    result = subject_ersp(raw, segments, sfreq, cycles, anchors, subject)
    if result is None:
        print("  No gait cycles survived -- skipping subject.")
        return
    weights = save_roi_weights(raw.info, dirs, subject)
    print_phase_summary(result, anchors)
    for key, path in outputs.items():
        np.save(path, result[key])
    np.save(out_beta, result["ersp"])
    print(f"  Saved -> {out_beta.name} and the double-stance / swing averages")
    log_ersp_qc(result, weights, dirs, subject)


def main() -> None:
    """Run stage 5 for one or all subjects."""
    subjects = subjects_from_command_line("TFR + ERSP (reference stage 5).")
    dirs = get_dataset_dirs(DATASET)
    anchors = load_reference_anchors(dirs["ersp"])  # frozen inputs, never rewritten here
    print(
        f"Reference anchors: LTO {anchors[0]:.2f} %  LHS {anchors[1]:.2f} %  RTO {anchors[2]:.2f} %"
    )
    run_for_subjects(subjects, lambda subject: process_subject(subject, dirs, anchors))


if __name__ == "__main__":
    main()
