"""
qc_multiverse_frontal.py -- heel-strike-locked frontal signal and ROI variants in every universe.

Purpose
    For each of the 27 universes (60 Hz branches), on the specification-curve
    cohort (treadmill n = 15, overground primary n = 14), from each universe's
    cached ICA-corrected data: the group heel-strike-locked 1-40 Hz
    peak-to-peak deflection (0-400 ms) at Fpz and Cz, and the double stance vs
    swing beta contrast (dB, t, p) with the linear Cz ROI, without Fp*/AF*,
    and for the five channels nearest Cz (src.multiverse_frontal). Then the
    correlation of each deflection with the stored ROI contrast across
    universes, overall and within each ASR mode. The universes share
    participants, so these correlations describe the multiverse and are not
    tests over independent samples.
Inputs
    results/multiverse/<ds>/branches/<subject>/<branch>/sub-*_desc-icaClean_<rule>_raw.fif
    results/multiverse/<ds>/outputs/<ds>_universe_subject_contrasts.tsv (cohort, stored values)
    results/pipeline/<ds>/qc/<ds>_multiverse_universes.tsv (stored universe statistics)
    datasets/<ds>/d01_gaitevents/sub-*_cycles.tsv, d05_ersp/group_gait_event_anchors_frozen.json
Outputs
    results/multiverse/<ds>/outputs/<ds>_multiverse_frontal.tsv
    results/pipeline/<ds>/qc/<ds>_multiverse_frontal.txt
Manuscript
    Discussion (frontal channels, movement artefact across the multiverse).
Run
    python scripts/qc/qc_multiverse_frontal.py <stepup|jacobsen> [n_workers]
    (default 4 workers; treadmill about 40 min, overground longer)
"""

import os
import sys

DATASET_ARG = sys.argv[1] if len(sys.argv) > 1 else ""
if DATASET_ARG not in ("stepup", "jacobsen"):
    raise SystemExit("Usage: qc_multiverse_frontal.py <stepup|jacobsen> [n_workers]")
os.environ["BETAGAIT_DATASET"] = DATASET_ARG

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DIR_MULTIVERSE_OUTPUTS, DIR_QC, MULTIVERSE_LOWPASS_HZ  # noqa: E402
from src.multiverse_frontal import correlation_lines, subject_universe, universe_row  # noqa: E402

KEYS = ["universe", "highpass_hz", "asr_mode", "iclabel_rule"]
STORED_COLUMNS = (("db", "mean_db"), ("t", "t"), ("p", "p"))
HEADER = (
    "# Values come from the float32 ICA-corrected caches (results/multiverse/{ds}/branches/); "
    "the stored universe results were computed from float64 data in memory, so per-subject "
    "ROI contrasts differ from the stored ones by up to {gap:.3f} dB. stored_roi_* are the "
    "stored values ({ds}_multiverse_universes.tsv).\n"
)


def cohort_jobs() -> tuple[pd.DataFrame, list[tuple]]:
    """Specification-curve cohort rows and one job per (universe, subject)."""
    path = DIR_MULTIVERSE_OUTPUTS / f"{DATASET_ARG}_universe_subject_contrasts.tsv"
    subj = pd.read_csv(path, sep="\t", dtype={"subject": str})  # keep "001", not 1
    subj = subj[subj["in_speccurve_cohort"]].reset_index(drop=True)
    jobs = []
    for _, r in subj.iterrows():
        decisions = {
            "highpass_hz": float(r["highpass_hz"]), "asr_mode": r["asr_mode"],
            "iclabel_rule": r["iclabel_rule"], "lowpass_hz": float(MULTIVERSE_LOWPASS_HZ),
        }  # fmt: skip
        jobs.append((DATASET_ARG, decisions, r["subject"]))
    return subj, jobs


def build_table(subj: pd.DataFrame, results: list[dict]) -> tuple[pd.DataFrame, float]:
    """Return the 27-row table (sorted by stored contrast) and the largest per-subject gap."""
    stored = pd.read_csv(DIR_QC / f"{DATASET_ARG}_multiverse_universes.tsv", sep="\t")
    stored = stored.set_index("universe")
    rows, gap = [], 0.0
    for key, g in subj.groupby(KEYS):
        row = universe_row([results[i] for i in g.index])
        gap = max(gap, float(np.abs(row.pop("roi_values") - g["beta_diff_db"].to_numpy()).max()))
        u = key[0]
        rows.append(
            dict(zip(KEYS, key, strict=True)) | row
            | {f"stored_roi_{k}": stored.loc[u, c] for k, c in STORED_COLUMNS}
        )  # fmt: skip
    table = pd.DataFrame(rows).sort_values("stored_roi_db").reset_index(drop=True)
    return table, gap


def main() -> None:
    """Compute every subject-universe in parallel, then write the table and the correlations."""
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    subj, jobs = cohort_jobs()
    print(f"{DATASET_ARG}: {len(jobs)} subject-universe jobs, {n_workers} workers", flush=True)
    with Pool(n_workers) as pool:
        results = pool.map(subject_universe, jobs, chunksize=1)
    table, gap = build_table(subj, results)
    out = DIR_MULTIVERSE_OUTPUTS / f"{DATASET_ARG}_multiverse_frontal.tsv"
    with open(out, "w", newline="\n") as f:
        f.write(HEADER.format(ds=DATASET_ARG, gap=gap))
        table.to_csv(f, sep="\t", index=False, float_format="%.4g", lineterminator="\n")
    n = int(table["n"].iloc[0])
    lines = [
        f"=== {DATASET_ARG}: 27 universes, n = {n} per universe (float32 caches; max per-subject "
        f"|cached - stored ROI contrast| = {gap:.3f} dB)",
        "Correlation of the heel-strike-locked peak-to-peak deflection with the stored ROI "
        "contrast.",
        "k = number of universes. The universes share participants and are not independent;",
        "p values are descriptive only.",
    ]
    lines += correlation_lines(table)
    lines += ["", "Universes with p >= 0.05: " + ", ".join(
        f"{k} {int((table[f'{k}_p'] >= 0.05).sum())}" for k in ("roi", "nofrontal", "narrow")
    )]  # fmt: skip
    (DIR_QC / f"{DATASET_ARG}_multiverse_frontal.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
