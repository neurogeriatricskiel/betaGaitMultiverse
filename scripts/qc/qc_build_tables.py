"""
qc_build_tables.py -- reviewer-facing QC tables for one dataset.

Purpose
    Collect the stored per-subject QC (d00_qc jsons, saved file headers),
    cohort definitions, exclusion decisions, the resolved configuration
    and the per-universe multiverse results into tracked files under
    results/pipeline/<ds>/qc/ (src.qc_tables). Reads only; recomputes nothing.
Inputs
    datasets/<ds>/d00_qc/sub-*_qc.json (incl. the treadmill S2/S21 recheck jsons)
    datasets/<ds>/d02_prep/sub-*_concat_raw.fif, sub-*_preica_clean_epo.fif (headers)
    datasets/<ds>/d03_clean/sub-*_desc-icaClean_concat_raw.fif (header)
    results/multiverse/<ds>/comet/results/multiverse_results.pkl (read-only)
    results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt (overground cohorts)
Outputs (results/pipeline/<ds>/qc/)
    <ds>_subject_qc.tsv            one row per subject: channels, ICA, AutoReject, cycles, flag
    <ds>_exclusions.tsv            subject, stage, reason, numbers, source
    <ds>_cohorts.json              reference, multiverse and (overground) primary cohorts
    <ds>_parameters_used.json      resolved src.config snapshot for this dataset
    <ds>_multiverse_universes.tsv  per universe: spec, mean dB, t, p, n, 95 % CI
    <ds>_qc_summary.tsv            every per-subject QC json, one row per subject and stage
    <ds>_qc_flags.tsv              QC flag per subject and stage
Manuscript
    Supplementary QC; Methods 2.1 (cohorts).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/qc/qc_build_tables.py
"""

import json

from src.config import DATASET, DIR_MULTIVERSE_COMET, DIR_QC
from src.multiverse_results import load_combined_results, measure_for, universe_table
from src.paths import get_dataset_dirs
from src.qc import write_qc_summary
from src.qc_tables import cohorts, exclusions, parameters_used, subject_table


def write_universe_table(path) -> None:
    """Per-universe statistics, the same numbers as the specification curve."""
    df = universe_table(load_combined_results(DIR_MULTIVERSE_COMET), measure_for(DATASET))
    df = df.rename(columns={"mean": "mean_db", "ci_lo": "ci95_lo_db", "ci_hi": "ci95_hi_db"})
    df.round(6).to_csv(path, sep="\t", index=False)


def write_qc_flag_tables(dirs: dict) -> None:
    """Long table of all per-subject QC records and a subject x stage flag table."""
    df = write_qc_summary(dirs["qc"], DIR_QC / f"{DATASET}_qc_summary.tsv")
    if df.empty:
        print("No QC records found. Run the pipeline first.")
        return
    flags = df.pivot_table(index="subject", columns="stage", values="flag", aggfunc="first")
    flags.reset_index().to_csv(DIR_QC / f"{DATASET}_qc_flags.tsv", sep="\t", index=False)
    problems = sorted(df[df["flag"] != "pass"]["subject"].unique())
    print(f"\n  Subjects with warn/fail flags: {problems}" if problems else "\n  All passed.")


def main() -> None:
    """Write every QC table of the active dataset."""
    dirs = get_dataset_dirs(DATASET)
    prefix = DIR_QC / DATASET
    subject_table(dirs).to_csv(f"{prefix}_subject_qc.tsv", sep="\t", index=False)
    exclusions(dirs).to_csv(f"{prefix}_exclusions.tsv", sep="\t", index=False)
    with open(f"{prefix}_cohorts.json", "w") as f:
        json.dump(cohorts(), f, indent=2)
    with open(f"{prefix}_parameters_used.json", "w") as f:
        json.dump(parameters_used(), f, indent=2, sort_keys=True)
    write_universe_table(f"{prefix}_multiverse_universes.tsv")
    write_qc_flag_tables(dirs)
    print(f"  Saved -> {prefix}_*.tsv / .json")


if __name__ == "__main__":
    main()
