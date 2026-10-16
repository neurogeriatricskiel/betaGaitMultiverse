"""
mulana04_zoom_universes.py -- multiverse, step 4: lowest / median / highest universe figure.

Purpose
    Rank the 27 universes by the group-mean double-stance - swing beta
    contrast (dB; treadmill "t_stats", n = 15; overground "dedup_diffs",
    primary cohort n = 14) and plot the group ROI ERSP (8-40 Hz) of the
    lowest, median (14th of 27) and highest universe with a beta trace
    under each heatmap (src.multiverse_zoom). Panel arrays are new
    computation from each universe's cached ICA-corrected data (never
    refits ICA, never writes under comet/), cached to
    results/multiverse/<ds>/outputs/<ds>_zoom_arrays.npz on the first run
    (40-45 min per dataset); delete the .npz to recompute. Each array's
    contrast must match the pkl within 1 % (float32 cache vs float64
    multiverse run) or the figure is not saved.
Inputs
    results/multiverse/<ds>/comet/results/multiverse_results.pkl (read-only)
    results/multiverse/<ds>/branches/<subject>/<branch>/sub-*_desc-icaClean_<rule>_raw.fif
    datasets/<ds>/d01_gaitevents/sub-*_cycles.tsv, d05_ersp/group_gait_event_anchors_frozen.json
Outputs
    results/figures/<ds>_multiverse_zoom.png
    results/multiverse/<ds>/outputs/<ds>_zoom_arrays.npz
Manuscript
    Results, multiverse zoom figure (Fig. treadmill / overground).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/mulana04_zoom_universes.py
"""

from pathlib import Path

import matplotlib
import numpy as np

from src.config import (
    DATASET,
    DIR_FIGURES,
    DIR_MULTIVERSE_COMET,
    DIR_MULTIVERSE_OUTPUTS,
    DIR_QC,
    MULTIVERSE_SUBJECTS,
)
from src.ersp import load_group_anchors
from src.multiverse_results import load_combined_results, measure_for
from src.multiverse_zoom import (
    check_contrasts,
    panel_arrays,
    print_selection,
    rank_universes,
    render_zoom_figure,
    select_universes,
)
from src.paths import get_dataset_dirs
from src.qc import load_primary_cohort

CACHE_PATH = Path(DIR_MULTIVERSE_OUTPUTS) / f"{DATASET}_zoom_arrays.npz"


def primary_cohort(all_universes: dict) -> list[str]:
    """Overground primary cohort, verified to reproduce "dedup_diffs" in every universe."""
    subjects = load_primary_cohort(DIR_QC)
    for key, data in all_universes.items():
        by_subject = dict(zip(data["subjects_ok"], data["t_stats"], strict=True))
        if not np.array_equal([by_subject[s] for s in subjects], data["dedup_diffs"]):
            raise SystemExit(f"{key}: PRIMARY cohort does not reproduce dedup_diffs")
    print(f"  PRIMARY cohort (n={len(subjects)}) reproduces dedup_diffs in all universes")
    return subjects


def main() -> None:
    """Rank universes, build or load the panel arrays and save the zoom figure."""
    matplotlib.use("Agg")
    if not (DIR_MULTIVERSE_COMET / "results" / "multiverse_results.pkl").exists():
        raise SystemExit("No multiverse results -- run mulana02_run_multiverse.py first.")
    all_universes = load_combined_results(DIR_MULTIVERSE_COMET)
    measure = measure_for(DATASET)
    # The pkl stores "dedup_diffs" values but not their subject IDs.
    subjects = primary_cohort(all_universes) if measure == "dedup_diffs" else MULTIVERSE_SUBJECTS

    table = rank_universes(all_universes, measure)
    selection = select_universes(table)
    print_selection(table, selection)

    dirs = get_dataset_dirs(DATASET)
    anchors = load_group_anchors(dirs["ersp"])
    group_ersp, n_used = panel_arrays(selection, subjects, dirs["gait_events"], anchors, CACHE_PATH)
    check_contrasts(selection, group_ersp, n_used, anchors)
    out_path = Path(DIR_FIGURES) / f"{DATASET}_multiverse_zoom.png"
    render_zoom_figure(selection, group_ersp, anchors, DATASET, out_path)


if __name__ == "__main__":
    main()
