"""
prepana07_betaphase_stats.py -- reference pipeline, stage 7: group beta statistics.

Purpose
    Reduce each subject's double-stance and swing ERSP (prepana05) to one
    beta (13-30 Hz) value with the linear Cz ROI weights, then test the
    per-subject double-stance - swing differences against 0 across
    subjects (paired t-test = one-sample t-test of the differences,
    two-sided, alpha 0.05; src.reference_stats). Runs on config.SUBJECTS
    (treadmill n = 17, overground n = 18); the overground primary n = 14
    statistic is in results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt.
Inputs
    datasets/<ds>/d05_ersp/sub-*_ersp_double_stance.npy, sub-*_ersp_swing.npy,
    sub-*_roi_weights.npy
Outputs
    results/pipeline/<ds>/qc/<ds>_betaphase_stats.txt
Manuscript
    Methods 2.8 (statistics); Results, reference statistics.
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/prepana07_betaphase_stats.py
"""

from src.config import DATASET, DIR_QC, SUBJECTS
from src.ersp import BETA_FMAX, BETA_FMIN
from src.paths import get_dataset_dirs
from src.reference_stats import paired_test, phase_beta_values


def subject_beta_values(subjects: list[str], ersp_dir) -> list[tuple[str, float, float]]:
    """Per-subject (subject, double-stance, swing) beta ROI values in dB."""
    rows = []
    for subject in subjects:
        values = phase_beta_values(subject, ersp_dir)
        if values is None:
            print(f"  [SKIP] sub-{subject}: missing double_stance/swing/roi_weights .npy")
            continue
        beta_ds, beta_sw = values
        rows.append((subject, beta_ds, beta_sw))
        print(
            f"  sub-{subject}: double_stance={beta_ds:+.3f} dB  swing={beta_sw:+.3f} dB  "
            f"diff={beta_ds - beta_sw:+.3f} dB"
        )
    return rows


def format_summary(rows: list[tuple[str, float, float]]) -> str:
    """Text of <dataset>_betaphase_stats.txt."""
    test = paired_test([r[1] for r in rows], [r[2] for r in rows])
    n = test["n"]
    lines = [
        "BETA PHASE GROUP STATISTICS -- double stance vs swing",
        "=" * 55,
        "Normalization: dB relative to standing baseline",
        f"Beta band: {BETA_FMIN:.0f}-{BETA_FMAX:.0f} Hz  (linear Cz ROI weights)",
        "",
        "Per-subject beta ROI-weighted ERSP (dB):",
        f"  {'subject':<10}{'double_stance':>15}{'swing':>12}{'diff':>12}",
    ]
    lines += [f"  {sub:<10}{ds:>15.3f}{sw:>12.3f}{ds - sw:>12.3f}" for sub, ds, sw in rows]
    lines += [
        "",
        f"Group paired t-test (double_stance - swing), n={n} subjects:",
        f"  t({n - 1}) = {test['t']:.3f}   p = {test['p']:.4g}",
        f"  mean diff = {test['mean_diff']:+.3f} dB",
        f"  subjects with double_stance > swing: {test['n_ds_gt_swing']}/{n}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Compute the group statistic and write the stats file."""
    rows = subject_beta_values(SUBJECTS, get_dataset_dirs(DATASET)["ersp"])
    if len(rows) < 2:
        raise RuntimeError(f"Only {len(rows)} subject(s) with double_stance/swing maps (need 2).")
    summary = format_summary(rows)
    print("\n" + summary)
    out_path = DIR_QC / f"{DATASET}_betaphase_stats.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary + "\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
