"""
qc_gpm_band_check.py -- overground GPM by beta sub-band (20-30 Hz band check).

Purpose
    Group-mean gait power modulation (GPM, src.gpm, as in prepana06b) over
    the gait cycle for 13-20, 20-30, 10-17 and 20-35 Hz (the last two match
    the source study's panels), at the linear Cz ROI and at Cz: peak and
    trough (value and % of cycle) and the mean in double stance minus the
    mean in swing, using the overground reference phase windows (fixed
    anchors 18/50/68 %). Shows whether 20-30 Hz is also higher in double
    stance. Also the paired t-test of ROI double stance vs swing per subject
    for 13-30 (the primary band), 13-20 and 20-30 Hz, FULL and PRIMARY cohorts.
Inputs
    datasets/jacobsen/d05_ersp/sub-*_ersp_beta.npy, sub-*_roi_weights.npy
    datasets/jacobsen/d03_clean/sub-*_desc-icaClean_concat_raw.fif (channel names)
    results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt (cohorts)
Outputs
    results/pipeline/jacobsen/qc/jacobsen_gpm_band_check.txt
Manuscript
    Discussion (beta sub-bands; comparison with the source study's GPM).
Run
    python scripts/qc/qc_gpm_band_check.py   (always overground; < 1 min)
"""

import os

os.environ["BETAGAIT_DATASET"] = "jacobsen"

import numpy as np  # noqa: E402
from scipy.stats import ttest_rel  # noqa: E402

from src.config import DIR_QC, ROI_CENTER_CH, TFR_N_POINTS  # noqa: E402
from src.ersp import load_reference_anchors, phase_split_indices  # noqa: E402
from src.gpm import subject_gpm  # noqa: E402
from src.paths import get_dataset_dirs  # noqa: E402
from src.qc import load_cohort_variants  # noqa: E402
from src.reference_ersp import FREQS  # noqa: E402

BANDS = [(13, 20), (20, 30), (10, 17), (20, 35)]
T_BANDS = [(13, 30), (13, 20), (20, 30)]


def band_profile_lines(groups, ds_idx, sw_idx) -> list[str]:
    """Peak, trough and double stance - swing of the group GPM per band and location."""
    lines = []
    for name, group in groups:
        lines.append(f"\n== {name} ==")
        for lo, hi in BANDS:
            y = group[(FREQS >= lo) & (FREQS <= hi)].mean(axis=0)
            lines.append(
                f"  {lo:>2}-{hi:<2} Hz: peak {y.max():+.3f} dB @ {y.argmax():3d}%   "
                f"trough {y.min():+.3f} dB @ {y.argmin():3d}%   "
                f"DS - swing {y[ds_idx].mean() - y[sw_idx].mean():+.3f} dB"
            )
    return lines


def band_test_lines(roi: dict, variants: dict, ds_idx, sw_idx) -> list[str]:
    """Per-subject ROI double stance vs swing t-test per band, FULL and PRIMARY."""
    lines = ["\n== ROI, per-subject double stance vs swing, paired t-test =="]
    for cohort in ("FULL", "PRIMARY"):
        for lo, hi in T_BANDS:
            band = (FREQS >= lo) & (FREQS <= hi)
            ds = np.array([roi[s][band][:, ds_idx].mean() for s in variants[cohort]])
            sw = np.array([roi[s][band][:, sw_idx].mean() for s in variants[cohort]])
            t, p = ttest_rel(ds, sw)
            d = ds - sw
            n = len(d)
            lines.append(
                f"  {cohort:<7s} n={n}  {lo:>2}-{hi:<2} Hz: t({n - 1}) = {t:.3f}  p = {p:.4g}  "
                f"mean = {d.mean():+.3f} dB  DS > swing {int((d > 0).sum())}/{n}"
            )
    return lines


def main() -> None:
    """Write the band profile and band t-tests of the overground GPM."""
    dirs = get_dataset_dirs("jacobsen")
    variants = load_cohort_variants(DIR_QC)
    subjects = variants["PRIMARY"]
    per_subject = {s: subject_gpm(s, dirs) for s in variants["FULL"]}
    roi = {s: v[0] for s, v in per_subject.items()}
    anchors = load_reference_anchors(dirs["ersp"])
    ds_idx, sw_idx = phase_split_indices(anchors, n_points=TFR_N_POINTS)
    groups = (
        ("ROI", np.mean([roi[s] for s in subjects], axis=0)),
        (ROI_CENTER_CH, np.mean([per_subject[s][1] for s in subjects], axis=0)),
    )
    lines = [
        f"Overground GPM by band, primary cohort n = {len(subjects)}; "
        f"double stance = 0-{anchors[0]:g} % and {anchors[1]:g}-{anchors[2]:g} %"
    ]
    lines += band_profile_lines(groups, ds_idx, sw_idx)
    lines += band_test_lines(roi, variants, ds_idx, sw_idx)
    text = "\n".join(lines)
    print(text)
    out = DIR_QC / "jacobsen_gpm_band_check.txt"
    out.write_text(text + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
