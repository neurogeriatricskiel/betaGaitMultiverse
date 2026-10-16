"""
qc_applied_settings.py -- applied-versus-configured check of the reference preprocessing.

Purpose
    For every saved subject, read high-pass, low-pass, sampling rate and the
    50 Hz notch (d02_prep raw), ICA method, variance and fit-data high-pass
    (d02_prep ICA) and the ASR flag and cutoff (preprocessing QC json),
    compare each with the resolved src.config (src.applied_settings) and
    write one row per subject and setting with a match flag.
Inputs
    datasets/<ds>/d02_prep/sub-*_concat_raw.fif, sub-*_ica.fif
    datasets/<ds>/d00_qc/sub-*_preprocessing_qc.json
Outputs
    results/pipeline/<ds>/qc/<ds>_applied_settings.tsv
Manuscript
    Methods 2.3-2.4 (reported settings are the applied ones).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/qc/qc_applied_settings.py   (< 2 min)
"""

import pandas as pd

from src.applied_settings import compare
from src.config import DATASET, DIR_QC, SUBJECTS
from src.paths import get_dataset_dirs


def main() -> None:
    """Compare configured and applied settings for every saved subject."""
    dirs = get_dataset_dirs(DATASET)
    rows = []
    for subject in SUBJECTS:
        if (dirs["prep"] / f"sub-{subject}_concat_raw.fif").exists():
            rows += compare(subject, dirs)
        else:
            print(f"  sub-{subject}: no saved preprocessed raw -- skipped")
    df = pd.DataFrame(rows)
    out = DIR_QC / f"{DATASET}_applied_settings.tsv"
    df.to_csv(out, sep="\t", index=False)

    bad = df[~df["match"]]
    print(f"{DATASET}: {df['subject'].nunique()} subjects, {len(df)} checks, {len(bad)} mismatches")
    if len(bad):
        print(bad.to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
