"""
mulana03_visualize_multiverse.py -- multiverse, step 3: specification curve (both datasets).

Purpose
    Per universe: mean per-subject double-stance - swing beta contrast
    (dB), one-sample t-test against 0 (green = p < 0.05, uncorrected for
    the 27 universes) and a t-based 95 % CI drawn as a per-universe error
    bar (src.multiverse_results.universe_table -- the same computation
    comet's specification_curve() used). Treadmill: "t_stats",
    MULTIVERSE_SUBJECTS (n = 15); overground: "dedup_diffs", primary
    cohort (n = 14). Drawn by src.spec_curve rather than comet, so row
    order and y-limits are fixed and nothing is written into comet/.
Inputs
    results/multiverse/<ds>/comet/results/multiverse_results.pkl (read-only)
Outputs
    results/figures/<ds>_speccurve.png
Manuscript
    Results, specification curves (Fig. treadmill / overground).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/mulana03_visualize_multiverse.py
"""

from pathlib import Path

import matplotlib
import pandas as pd

from src import plot_style as ps
from src.config import ALPHA, DATASET, DIR_FIGURES, DIR_MULTIVERSE_COMET
from src.multiverse_results import load_combined_results, measure_for, universe_table
from src.spec_curve import CI_LEVEL, check_ci_within_ylim, plot_spec_curve

PRINT_COLUMNS = [
    "universe",
    "highpass_hz",
    "asr_mode",
    "iclabel_rule",
    "mean",
    "t",
    "p",
    "ci_lo",
    "ci_hi",
]


def main() -> None:
    """Build the per-universe table and save this dataset's specification curve."""
    matplotlib.use("Agg")
    measure = measure_for(DATASET)
    table = universe_table(load_combined_results(DIR_MULTIVERSE_COMET), measure, CI_LEVEL)
    n_values = sorted(table["n"].unique())
    if len(n_values) != 1:
        raise SystemExit(f"Universes differ in n: {n_values}")
    ylim = ps.SPEC_YLIM[DATASET]

    print(f"  {DATASET}: {len(table)} universes, n = {n_values[0]}, measure '{measure}'")
    print(f"  {int((table['p'] < ALPHA).sum())}/{len(table)} universes p < {ALPHA} (uncorrected)")
    print(
        f"  CI range {table['ci_lo'].min():+.3f} .. {table['ci_hi'].max():+.3f} dB, "
        f"fixed y-limits {ylim}"
    )
    check_ci_within_ylim(table, ylim)
    with pd.option_context("display.width", 160, "display.max_rows", 50):
        print(table[PRINT_COLUMNS].round(4).to_string(index=False))

    plot_spec_curve(
        table,
        DATASET,
        "specification curve, 27 universes",
        int(n_values[0]),
        ylim,
        Path(DIR_FIGURES) / f"{DATASET}_speccurve.png",
    )


if __name__ == "__main__":
    main()
