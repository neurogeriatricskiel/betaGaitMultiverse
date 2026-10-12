"""
qc_summary.py -- aggregate the per-subject QC jsons into group tables.

Purpose
    Collect every stage's sub-*_qc.json (written by prepana01-05) into one
    long table and one wide flag table (subject x stage).
Inputs
    datasets/<ds>/d00_qc/sub-*_qc.json
Outputs
    results/pipeline/<ds>/qc/<ds>_qc_summary.tsv  (one row per subject per stage)
    results/pipeline/<ds>/qc/<ds>_qc_flags.tsv    (one row per subject, flag per stage)
Manuscript
    Supplementary QC tables.
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/qc/qc_summary.py
"""

from src.config import DATASET, DIR_QC
from src.paths import get_dataset_dirs
from src.qc import write_qc_summary


def main():
    df = write_qc_summary(get_dataset_dirs(DATASET)["qc"], DIR_QC / f"{DATASET}_qc_summary.tsv")
    if df.empty:
        print("No QC records found. Run the pipeline first.")
        return

    flags = df.pivot_table(
        index="subject", columns="stage", values="flag", aggfunc="first"
    ).reset_index()
    flags_path = DIR_QC / f"{DATASET}_qc_flags.tsv"
    flags.to_csv(flags_path, sep="\t", index=False)
    print(f"  Flag table -> {flags_path}")

    problem_subjects = df[df["flag"] != "pass"]["subject"].unique()
    if len(problem_subjects):
        print(f"\n  Subjects with warn/fail flags: {sorted(problem_subjects)}")
    else:
        print("\n  All subjects passed all QC stages.")


if __name__ == "__main__":
    main()
