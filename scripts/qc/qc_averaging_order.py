"""
qc_averaging_order.py -- ERSP (and overground GPM) under our vs the source order of averaging.

Purpose
    Ours  : ERSP = mean_k 10*log10(P_k / P_standing)   (prepana05)
    Source: ERSP = 20*log10(mean_k |X_k| / mean |X_standing|), |X| = sqrt(P)
    Recomputes the TFR per cycle exactly as prepana05 (src.averaging_order;
    max |recomputed - stored| per subject is printed), reduces with the
    linear Cz ROI and reports, for the overground FULL and PRIMARY cohorts
    (treadmill: the reference cohort, beta ERSP only):
    - GPM (cycle-mean removed, src.gpm): pixel and beta-trace correlation,
      amplitude ratio, plus the double stance vs swing t-test per order;
    - beta ERSP (absolute level kept): range, mean, % negative pixels and
      maximum of the group map, subjects whose ROI-mean beta ERSP changes
      sign, and the double stance vs swing t-test per order.
Inputs
    datasets/<ds>/d03_clean/, d04_gaitepochs/, d05_ersp/
    results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt (overground cohorts)
Outputs
    results/pipeline/jacobsen/qc/jacobsen_gpm_averaging_order.txt, .npz, .png
    results/pipeline/<ds>/qc/<ds>_ersp_averaging_order.txt
Manuscript
    Discussion (ERD level and GPM vs the source study).
Run
    python scripts/qc/qc_averaging_order.py [jacobsen|stepup]
    (default overground; ~25 min per dataset, TFR of all cycles)
"""

import os
import sys

DATASET_ARG = sys.argv[1] if len(sys.argv) > 1 else "jacobsen"
if DATASET_ARG not in ("jacobsen", "stepup"):
    raise SystemExit(f"Unknown dataset {DATASET_ARG!r}; use jacobsen or stepup.")
os.environ["BETAGAIT_DATASET"] = DATASET_ARG

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

from src.averaging_order import (  # noqa: E402
    BETA_MASK,
    ERSP_COHORTS,
    ersp_map_lines,
    gpm_summary_lines,
    order_contrast_lines,
    save_gpm_order_plot,
    sign_change_lines,
    subject_ersp_orders,
)
from src.config import DIR_QC, SUBJECTS, TFR_N_POINTS  # noqa: E402
from src.ersp import load_reference_anchors, phase_split_indices  # noqa: E402
from src.gpm import gait_power_modulation  # noqa: E402
from src.paths import get_dataset_dirs  # noqa: E402
from src.qc import load_cohort_variants  # noqa: E402
from src.spatial_filter import apply_linear_roi  # noqa: E402

ERSP_HEADER = [
    "Overground beta ERSP: ours = mean of per-cycle dB (stored ERSP); "
    "source = dB of mean amplitude (Jacobsen et al.)",
    "",
]
STEPUP_HEADER = [
    "Treadmill beta ERSP: ours = mean of per-cycle dB (stored ERSP); "
    "source = dB of mean amplitude (Jacobsen et al.'s order)",
    "",
]


def subject_views(subject: str, dirs: dict, anchors) -> tuple[tuple, tuple]:
    """ROI GPM and ROI beta ERSP of one subject, each as (ours, source)."""
    stored, source, weights = subject_ersp_orders(subject, dirs, anchors)
    gpm = tuple(apply_linear_roi(gait_power_modulation(e), weights) for e in (stored, source))
    beta = tuple(apply_linear_roi(e, weights)[BETA_MASK] for e in (stored, source))
    return gpm, beta


def write_text(name: str, lines: list[str]) -> None:
    """Print the report and write it to the active dataset's QC folder."""
    print("\n" + "\n".join(lines))
    (DIR_QC / name).write_text("\n".join(lines) + "\n")


def stepup_main() -> None:
    """Treadmill reference cohort (subjects with a stage-5 ERSP): beta ERSP report only."""
    dirs = get_dataset_dirs("stepup")
    anchors = load_reference_anchors(dirs["ersp"])
    cohort = [s for s in SUBJECTS if (dirs["ersp"] / f"sub-{s}_ersp_beta.npy").exists()]
    variants = {"REFERENCE": cohort}
    beta = {s: subject_views(s, dirs, anchors)[1] for s in cohort}
    phase_idx = phase_split_indices(anchors, n_points=TFR_N_POINTS)
    lines = STEPUP_HEADER + ersp_map_lines(beta, variants, ("REFERENCE",))
    lines += sign_change_lines(beta, variants, ("REFERENCE",), "REFERENCE")
    lines += order_contrast_lines(beta, variants, ("REFERENCE",), phase_idx, True)
    write_text("stepup_ersp_averaging_order.txt", lines)


def main() -> None:
    """Recompute both orders for the full cohort and write the GPM and ERSP reports."""
    matplotlib.use("Agg")
    if DATASET_ARG == "stepup":
        stepup_main()
        return
    dirs = get_dataset_dirs("jacobsen")
    anchors = load_reference_anchors(dirs["ersp"])
    variants = load_cohort_variants(DIR_QC)
    views = {s: subject_views(s, dirs, anchors) for s in variants["FULL"]}
    gpm = {s: v[0] for s, v in views.items()}
    beta = {s: v[1] for s, v in views.items()}
    phase_idx = phase_split_indices(anchors, n_points=TFR_N_POINTS)

    primary = variants["PRIMARY"]
    ours = np.mean([gpm[s][0] for s in primary], 0)
    source = np.mean([gpm[s][1] for s in primary], 0)
    gpm_lines = gpm_summary_lines(ours, source, len(primary))
    gpm_beta = {s: tuple(g[BETA_MASK] for g in v) for s, v in gpm.items()}
    gpm_lines += order_contrast_lines(gpm_beta, variants, ("FULL", "PRIMARY"), phase_idx, False)
    write_text("jacobsen_gpm_averaging_order.txt", gpm_lines)
    np.savez(DIR_QC / "jacobsen_gpm_averaging_order.npz", ours=ours, source=source)
    save_gpm_order_plot(ours, source, anchors, DIR_QC / "jacobsen_gpm_averaging_order.png")

    ersp_lines = ERSP_HEADER + ersp_map_lines(beta, variants) + sign_change_lines(beta, variants)
    ersp_lines += order_contrast_lines(beta, variants, ERSP_COHORTS, phase_idx, True)
    write_text("jacobsen_ersp_averaging_order.txt", ersp_lines)


if __name__ == "__main__":
    main()
