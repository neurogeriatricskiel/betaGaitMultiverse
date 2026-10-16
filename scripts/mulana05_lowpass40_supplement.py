"""
mulana05_lowpass40_supplement.py -- treadmill multiverse at a 40 Hz vs 60 Hz low-pass (supplement).

Purpose
    The treadmill multiverse was first run with a 40 Hz raw low-pass; the
    reported run uses 60 Hz. Same 27-universe grid, same per-subject
    double-stance - swing beta contrast ("t_stats"). Per universe: mean
    (dB), one-sample t, p and n at 60 Hz and at 40 Hz
    (src.multiverse_results.universe_table), and the 40 Hz specification
    curve drawn with the main figure's code (src.spec_curve). Checks that
    the 60 Hz columns equal the current specification-curve statistics.
    Read-only on both pkl files.
Inputs
    results/multiverse/stepup/comet/results/multiverse_results.pkl
    results/multiverse/stepup_40hz_archive/comet/results/multiverse_results.pkl
Outputs
    results/supplementary/stepup_multiverse_lowpass40_vs_60.tsv
    results/supplementary/stepup_speccurve_lowpass40.png
Manuscript
    Supplement (sensitivity of the treadmill multiverse to the low-pass).
Run
    python scripts/mulana05_lowpass40_supplement.py   (always treadmill; < 1 min)
"""

import os

os.environ["BETAGAIT_DATASET"] = "stepup"

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import plot_style as ps  # noqa: E402
from src.config import (  # noqa: E402
    ARCHIVE_LOWPASS_HZ,
    DATASET,
    DIR_MULTIVERSE_COMET,
    DIR_MULTIVERSE_LOWPASS40_ARCHIVE,
    DIR_QC,
    DIR_SUPPLEMENTARY,
    MULTIVERSE_LOWPASS_HZ,
)
from src.multiverse_results import load_combined_results, measure_for, universe_table  # noqa: E402
from src.spec_curve import CI_LEVEL, check_ci_within_ylim, plot_spec_curve  # noqa: E402

DECISIONS = ["highpass_hz", "asr_mode", "iclabel_rule"]
STATS = ["mean", "t", "p", "n"]


def lowpass_table(comet_dir: Path) -> pd.DataFrame:
    """Per-universe statistics of one multiverse run, indexed by universe number.

    Parameters
    ----------
    comet_dir : pathlib.Path
        COMET working directory holding ``results/multiverse_results.pkl``.

    Returns
    -------
    pandas.DataFrame
        Decisions, mean, t, p, n and CI bounds per universe.
    """
    table = universe_table(load_combined_results(comet_dir), measure_for(DATASET), CI_LEVEL)
    return table.set_index("universe").sort_index()


def check_60hz_matches_current(at60: pd.DataFrame) -> None:
    """Raise SystemExit unless the 60 Hz statistics equal the QC per-universe table.

    Parameters
    ----------
    at60 : pandas.DataFrame
        Output of ``lowpass_table`` for the current (60 Hz) run.
    """
    qc = pd.read_csv(DIR_QC / f"{DATASET}_multiverse_universes.tsv", sep="\t")
    qc = qc.set_index("universe").sort_index()
    for ours, theirs in (("mean", "mean_db"), ("t", "t"), ("p", "p"), ("n", "n")):
        if not np.allclose(at60[ours], qc[theirs], rtol=0, atol=5e-6):
            raise SystemExit(f"60 Hz column {ours} differs from {DATASET}_multiverse_universes.tsv")
    print("  60 Hz columns equal the current specification-curve statistics.")


def main() -> None:
    """Write the 40 vs 60 Hz table and the 40 Hz specification curve."""
    matplotlib.use("Agg")
    at60 = lowpass_table(DIR_MULTIVERSE_COMET)
    at40 = lowpass_table(DIR_MULTIVERSE_LOWPASS40_ARCHIVE / "comet")
    if not at60[DECISIONS].astype(str).equals(at40[DECISIONS].astype(str)):
        raise SystemExit("Universe numbers map to different decisions in the two runs.")
    check_60hz_matches_current(at60)

    lp60, lp40 = f"{MULTIVERSE_LOWPASS_HZ:g}hz", f"{ARCHIVE_LOWPASS_HZ:g}hz"
    out = at60[DECISIONS].copy()
    for label, table in ((lp60, at60), (lp40, at40)):
        for stat in STATS:
            out[f"{'mean_db' if stat == 'mean' else stat}_lowpass{label}"] = table[stat]
    tsv = Path(DIR_SUPPLEMENTARY) / f"{DATASET}_multiverse_lowpass40_vs_60.tsv"
    out.reset_index().to_csv(tsv, sep="\t", index=False, float_format="%.6g", lineterminator="\n")
    print(f"  Saved -> {tsv}")
    n_counts = {int(k): int(v) for k, v in at40["n"].value_counts().sort_index().items()}
    print(f"  40 Hz n per universe (n: universes): {n_counts}")

    ylim = ps.SPEC_YLIM[DATASET]
    table40 = at40.reset_index().sort_values("mean", kind="stable").reset_index(drop=True)
    check_ci_within_ylim(table40, ylim)
    n_values = sorted(int(n) for n in table40["n"].unique())
    n_label = n_values[0] if len(n_values) == 1 else f"{n_values[0]}-{n_values[-1]}"
    plot_spec_curve(
        table40,
        DATASET,
        f"specification curve, 27 universes, {ARCHIVE_LOWPASS_HZ:g} Hz low-pass",
        n_label,
        ylim,
        Path(DIR_SUPPLEMENTARY) / f"{DATASET}_speccurve_lowpass40.png",
    )


if __name__ == "__main__":
    main()
