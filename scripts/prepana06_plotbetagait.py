"""
prepana06_plotbetagait.py -- reference pipeline, stage 6: group ERSP figure.

Purpose
    Group-average ROI ERSP (TFR_FMIN-PIPELINE_TFR_FMAX, 8-60 Hz) over the
    normalised gait cycle with the beta (13-30 Hz) trace, gait events and
    phase bar, plus beta topographies for the whole cycle, swing and
    double stance. Treadmill cohort: every subject with an ERSP file
    (n = 17); overground: the primary independent-recordings cohort
    (n = 14, src.qc.load_primary_cohort, the source of the primary
    statistic). Prints the double-stance - swing t-test on the displayed
    cohort as a check (src.reference_figure).
Inputs
    datasets/<ds>/d05_ersp/sub-*_ersp_beta.npy, sub-*_roi_weights.npy
    datasets/<ds>/d04_gaitepochs/sub-*_cycles_kept.tsv
    datasets/<ds>/d03_clean/sub-*_desc-icaClean_concat_raw.fif (channel info)
Outputs
    results/figures/<ds>_betaersp_gait.png
Manuscript
    Results, reference ERSP figure (Fig. treadmill / overground).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/prepana06_plotbetagait.py
"""

from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import ttest_rel

from src.config import DATASET, DIR_FIGURES, DIR_QC, SUBJECTS
from src.ersp import load_reference_anchors
from src.paths import get_dataset_dirs
from src.qc import load_primary_cohort
from src.reference_figure import group_topographies, load_group, render_reference_figure


def main() -> None:
    """Load the cohort's ERSP, print the displayed-cohort statistic and save the figure."""
    matplotlib.use("Agg")
    dirs = get_dataset_dirs(DATASET)
    cohort = load_primary_cohort(DIR_QC) if DATASET == "jacobsen" else SUBJECTS
    # Anchors the ERSP arrays were warped to (prepana05), so event lines match.
    anchors = load_reference_anchors(dirs["ersp"])
    g = load_group(cohort, dirs, anchors)
    n_subjects = len(g["ersp"])
    if n_subjects == 0:
        raise SystemExit("No subjects loaded.")
    ersp_group = np.mean(np.stack(g["ersp"]), axis=0)  # (n_freqs, 101)
    print(f"\n  Group average: n={n_subjects} subjects")
    print(f"  Group ERSP range: {ersp_group.min():.2f} / {ersp_group.max():.2f} dB")
    print(f"  Group events: LTO={anchors[0]:.1f}%  LHS={anchors[1]:.1f}%  RTO={anchors[2]:.1f}%")
    # Check: the displayed cohort reproduces the group statistic.
    t_stat, p_val = ttest_rel(g["roi_ds"], g["roi_sw"])
    print(
        f"  DS - swing (ROI, 13-30 Hz) on the displayed cohort: n={n_subjects}  "
        f"t({n_subjects - 1}) = {t_stat:.3f}  p = {p_val:.4g}  "
        f"mean diff = {np.mean(np.subtract(g['roi_ds'], g['roi_sw'])):+.3f} dB"
    )
    out_path = Path(DIR_FIGURES) / f"{DATASET}_betaersp_gait.png"
    render_reference_figure(
        ersp_group, group_topographies(g), g["info"], anchors, n_subjects, DATASET, out_path
    )


if __name__ == "__main__":
    main()
