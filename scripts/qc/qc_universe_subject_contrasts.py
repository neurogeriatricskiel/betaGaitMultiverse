"""
qc_universe_subject_contrasts.py -- per-universe, per-subject multiverse contrasts as a table.

Purpose
    Flattens the stored multiverse results into one row per universe and
    subject: decisions, subject, beta (13-30 Hz) double stance, swing and
    their difference (dB), and cohort flags. Then recomputes the
    specification-curve statistics (n, mean, t, p, CI per universe) from the
    table and checks they equal src.multiverse_results.universe_table() on
    the pkl. Read-only on the pkl; nothing is recomputed from EEG.
    Treadmill: the pkl stores only the per-subject differences ("t_stats",
    in MULTIVERSE_SUBJECTS order), so double stance and swing are empty.
    Overground: "subjects_ok", "beta_ds", "beta_sw" for all 18 subjects;
    the specification curve uses the primary cohort ("dedup_diffs").
Inputs
    results/multiverse/<ds>/comet/results/multiverse_results.pkl (read-only)
    results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt (primary cohort)
Outputs
    results/multiverse/<ds>/outputs/<ds>_universe_subject_contrasts.tsv
Manuscript
    Supplementary data behind the specification curves.
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/qc/qc_universe_subject_contrasts.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATASET,
    DIR_MULTIVERSE_COMET,
    DIR_MULTIVERSE_OUTPUTS,
    DIR_QC,
    MULTIVERSE_SUBJECTS,
)
from src.multiverse_results import (
    contrast_statistics,
    decisions_dict,
    load_combined_results,
    measure_for,
    universe_table,
)
from src.qc import load_primary_cohort

CI_LEVEL = 95
STAT_COLUMNS = ["n", "mean", "t", "p", "ci_lo", "ci_hi"]


def universe_rows(key: str, data: dict, speccurve_cohort: set) -> list[dict]:
    """Rows of one universe; checks the pkl's per-subject fields against each other."""
    decisions = decisions_dict(data["__decisions"])
    diffs = np.asarray(data["t_stats"], dtype=float)
    if "subjects_ok" in data:
        subjects = list(data["subjects_ok"])
        ds = np.asarray(data["beta_ds"], dtype=float)
        sw = np.asarray(data["beta_sw"], dtype=float)
        if not np.allclose(ds - sw, diffs, rtol=0, atol=1e-12):
            raise SystemExit(f"{key}: beta_ds - beta_sw differs from t_stats")
    else:
        if data["n_subjects_ok"] != len(MULTIVERSE_SUBJECTS):
            raise SystemExit(f"{key}: {data['n_subjects_ok']} subjects ok, cannot map to IDs")
        subjects = list(MULTIVERSE_SUBJECTS)
        ds = sw = np.full(len(subjects), np.nan)
    if len(subjects) != diffs.size:
        raise SystemExit(f"{key}: {len(subjects)} subjects vs {diffs.size} contrasts")
    if "dedup_diffs" in data:
        primary = [i for i, s in enumerate(subjects) if s in speccurve_cohort]
        if not np.array_equal(diffs[primary], np.asarray(data["dedup_diffs"], dtype=float)):
            raise SystemExit(f"{key}: dedup_diffs differ from t_stats on the primary cohort")
    return [
        {
            "universe": int(key.split("_")[1]),
            "highpass_hz": decisions["highpass_hz"],
            "asr_mode": decisions["asr_mode"],
            "iclabel_rule": decisions["iclabel_rule"],
            "subject": s,
            "beta_double_stance_db": ds[i],
            "beta_swing_db": sw[i],
            "beta_diff_db": diffs[i],
            "in_multiverse_cohort": s in MULTIVERSE_SUBJECTS,
            "in_speccurve_cohort": s in speccurve_cohort,
        }
        for i, s in enumerate(subjects)
    ]


def stats_from_table(table: pd.DataFrame) -> pd.DataFrame:
    """Specification-curve statistics per universe, from the table's speccurve rows."""
    rows = table[table["in_speccurve_cohort"]].groupby("universe")["beta_diff_db"]
    records = {u: contrast_statistics(g.to_numpy(dtype=float), CI_LEVEL) for u, g in rows}
    return pd.DataFrame.from_dict(records, orient="index").sort_index()


def main() -> None:
    """Write the table and check it reproduces the specification-curve statistics."""
    results = load_combined_results(DIR_MULTIVERSE_COMET)
    speccurve_cohort = (
        set(load_primary_cohort(DIR_QC)) if DATASET == "jacobsen" else set(MULTIVERSE_SUBJECTS)
    )
    table = pd.DataFrame(
        [r for key, data in results.items() for r in universe_rows(key, data, speccurve_cohort)]
    ).sort_values(["universe", "subject"], kind="stable")

    out = Path(DIR_MULTIVERSE_OUTPUTS) / f"{DATASET}_universe_subject_contrasts.tsv"
    table.to_csv(out, sep="\t", index=False, lineterminator="\n")  # full float precision
    print(f"  Saved -> {out}  ({out.stat().st_size:,} bytes, {len(table)} rows)")

    # Check: statistics recomputed from the written file equal the specification curve's.
    current = (
        universe_table(results, measure_for(DATASET), ci_level=CI_LEVEL)
        .set_index("universe")
        .sort_index()[STAT_COLUMNS]
    )
    recomputed = stats_from_table(pd.read_csv(out, sep="\t", dtype={"subject": str}))
    recomputed = recomputed[STAT_COLUMNS]
    max_abs = (recomputed - current).abs().max()
    print(
        "  max |file - pkl| per statistic: " + "  ".join(f"{k} {v:.1e}" for k, v in max_abs.items())
    )
    if not np.allclose(recomputed.to_numpy(float), current.to_numpy(float), rtol=1e-12, atol=0):
        out.unlink()
        raise SystemExit("  MISMATCH: statistics from the file differ from the spec curve.")
    print("  specification-curve statistics reproduced from the file.")


if __name__ == "__main__":
    main()
