"""
prepana06b_plot_gpm_supplement.py -- supplementary figure: overground gait power modulation.

Purpose
    Display-only comparison with the source study's GPM panel (Jacobsen
    et al.): GPM(f, t) = ERSP(f, t) - mean_t ERSP(f, t), per subject and
    channel, then reduced with the linear Cz ROI weights (left) or taken
    at Cz (right) and averaged over the primary cohort (n = 14; src.gpm).
    Reuses the stored prepana05 ERSP; nothing is recomputed or saved besides
    the figure. Because GPM subtracts a per-(channel, frequency) constant,
    the double-stance - swing contrast is unchanged: the script recomputes
    it from the GPM arrays and refuses to save the figure unless it equals
    the primary result.
Inputs
    datasets/jacobsen/d05_ersp/sub-*_ersp_beta.npy, sub-*_roi_weights.npy
    datasets/jacobsen/d03_clean/sub-*_desc-icaClean_concat_raw.fif (channel names)
    results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt (primary cohort)
Outputs
    results/supplementary/jacobsen_gpm_gait.png
Manuscript
    Supplementary figure (GPM); Discussion, comparison with the source study.
Run
    python scripts/prepana06b_plot_gpm_supplement.py   (always overground)
"""

import os

# Overground-only figure: fix the dataset before src.config is imported.
os.environ["BETAGAIT_DATASET"] = "jacobsen"

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import ttest_rel  # noqa: E402

from src.config import DATASET, DIR_QC, DIR_SUPPLEMENTARY, SUBJECTS, TFR_N_POINTS  # noqa: E402
from src.ersp import load_reference_anchors, phase_split_indices  # noqa: E402
from src.gpm import BETA_MASK, render_gpm_figure, subject_gpm  # noqa: E402
from src.paths import get_dataset_dirs  # noqa: E402
from src.qc import load_cohort_stats, load_primary_cohort  # noqa: E402


def check_contrast_unchanged(gpm_roi: list, anchors, n_subjects: int) -> None:
    """Stop unless the GPM double-stance - swing contrast equals the primary result."""
    expected_t, expected_p, expected_diff = load_cohort_stats(DIR_QC)["PRIMARY"]
    ds_idx, sw_idx = phase_split_indices(anchors, n_points=TFR_N_POINTS)
    ds = np.array([float(g[BETA_MASK][:, ds_idx].mean()) for g in gpm_roi])
    sw = np.array([float(g[BETA_MASK][:, sw_idx].mean()) for g in gpm_roi])
    t_stat, p_val = ttest_rel(ds, sw)
    mean_diff = float((ds - sw).mean())
    print(
        f"Check 2 -- DS - swing from GPM arrays: n={n_subjects}  t({n_subjects - 1}) = "
        f"{t_stat:.3f}  p = {p_val:.4f}  mean diff = {mean_diff:+.3f} dB"
    )
    if not (
        n_subjects == 14
        and round(t_stat, 3) == expected_t
        and np.isclose(p_val, expected_p, rtol=1e-3)  # stored to 4 significant digits
        and round(mean_diff, 3) == expected_diff
    ):
        raise SystemExit(
            f"  MISMATCH vs primary result t(13)={expected_t}, p={expected_p}, "
            f"+{expected_diff} dB -- figure NOT saved."
        )
    print("  matches primary result.")


def main() -> None:
    """Build the primary-cohort GPM, check the contrast and save the figure."""
    matplotlib.use("Agg")
    dirs = get_dataset_dirs(DATASET)
    subjects = load_primary_cohort(DIR_QC)
    unknown = set(subjects) - set(SUBJECTS)
    if unknown:
        raise RuntimeError(f"PRIMARY cohort has subjects outside SUBJECTS: {unknown}")
    print(f"PRIMARY cohort ({len(subjects)}): {','.join(subjects)}")
    anchors = load_reference_anchors(dirs["ersp"])
    roi, cz = zip(*(subject_gpm(s, dirs) for s in subjects), strict=True)
    roi_group, cz_group = np.mean(np.stack(roi), axis=0), np.mean(np.stack(cz), axis=0)
    print(
        f"\nCheck 1 -- max |row mean over cycle|: ROI {np.abs(roi_group.mean(axis=1)).max():.2e}"
        f" dB   Cz {np.abs(cz_group.mean(axis=1)).max():.2e} dB"
    )
    check_contrast_unchanged(list(roi), anchors, len(subjects))
    panels = [("Linear ROI centred on Cz", roi_group), ("Single channel Cz", cz_group)]
    out_path = Path(DIR_SUPPLEMENTARY) / f"{DATASET}_gpm_gait.png"
    render_gpm_figure(panels, anchors, len(subjects), DATASET, out_path)


if __name__ == "__main__":
    main()
