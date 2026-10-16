"""Per-subject QC records, stage QC flags and the overground cohort loaders.

Each reference stage writes one JSON record per subject with log_qc()
(scalar metrics and a pass / warn / fail flag); write_qc_summary() collects
them into one TSV. Flags: pass = all metrics in range; warn = a metric at
the boundary (usable, inspect); fail = a critical metric out of range.
The overground cohorts (FULL n = 18, PRIMARY n = 14, MINIMAL n = 16) are
read from jacobsen_duplicate_recordings.txt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DUPLICATE_REPORT = "jacobsen_duplicate_recordings.txt"
STAGE_ORDER = ["gait_events", "preprocessing", "ica", "gait_epochs", "ersp"]

# Stage QC thresholds (flags only; no subject is dropped automatically).
GAIT_MIN_CYCLES_FAIL, GAIT_MIN_CYCLES_WARN = 30, 60
GAIT_MAX_DURATION_CV = 0.15  # cycle-duration coefficient of variation (Hausdorff et al. 2001)
PREP_MIN_COMPONENTS = 10
MIN_BRAIN_ICS_WARN = 3
ICA_MIN_BRAIN_FRACTION = 0.15
EPOCHS_MIN_FAIL, EPOCHS_MIN_WARN, EPOCHS_MAX_REJECT_FRACTION = 20, 50, 0.3
ERSP_MIN_CYCLES_FAIL, ERSP_MIN_CYCLES_WARN, ERSP_MIN_RANGE_DB = 20, 50, 1.0
BASELINE_POWER_RANGE_V2 = (1e-13, 1e-9)  # plausible mean standing power


def load_cohort_variants(qc_dir: Path | str) -> dict[str, list[str]]:
    """Read the overground cohorts from jacobsen_duplicate_recordings.txt.

    Parameters
    ----------
    qc_dir : path-like
        Overground QC folder.

    Returns
    -------
    dict
        {"FULL": [...], "PRIMARY": [...], "MINIMAL": [...]}, subject IDs from
        the first "(a) FULL", "(b) PRIMARY" and "(c) MINIMAL" table rows.
    """
    variants = {}
    for line in (Path(qc_dir) / DUPLICATE_REPORT).read_text().splitlines():
        m = re.match(r"\s*\((a|b|c)\) (FULL|PRIMARY|MINIMAL)\b", line)
        if m and m.group(2) not in variants and "," in line.split()[-1]:
            variants[m.group(2)] = line.split()[-1].split(",")
    return variants


def load_cohort_stats(qc_dir: Path | str) -> dict[str, tuple[float, float, float]]:
    """Read the stored reference statistics of the overground cohorts.

    Parameters
    ----------
    qc_dir : path-like
        Overground QC folder.

    Returns
    -------
    dict
        {"FULL" / "PRIMARY" / "MINIMAL": (t, p, mean difference in dB)}.
    """
    out = {}
    for line in (Path(qc_dir) / DUPLICATE_REPORT).read_text().splitlines():
        m = re.match(
            r"\s*\((a|b|c)\) (FULL|PRIMARY|MINIMAL).*?\s(\d+)\s+(\d+)\s+([-\d.]+)\s+"
            r"([\d.e-]+)\s+([+-][\d.]+) dB",
            line,
        )
        if m and m.group(2) not in out:
            out[m.group(2)] = (float(m.group(5)), float(m.group(6)), float(m.group(7)))
    return out


def load_primary_cohort(qc_dir: Path | str) -> list[str]:
    """Return the overground PRIMARY independent-recordings cohort (n = 14).

    Parameters
    ----------
    qc_dir : path-like
        Overground QC folder.

    Returns
    -------
    list of str
        Subject IDs; the cohort of the primary statistic and of the
        multiverse pkl's "dedup_diffs" field.
    """
    variants = load_cohort_variants(qc_dir)
    if "PRIMARY" not in variants:
        raise RuntimeError(f"No '(b) PRIMARY' row found in {Path(qc_dir) / DUPLICATE_REPORT}")
    return variants["PRIMARY"]


def gait_events_qc(cycles_df: pd.DataFrame) -> tuple[str, dict]:
    """Return the stage-1 flag and metrics (cycle count and duration variability).

    Parameters
    ----------
    cycles_df : pandas.DataFrame
        Valid gait cycles with a duration_s column.

    Returns
    -------
    flag : str
        "pass", "warn" or "fail".
    metrics : dict
        n_cycles, mean_dur_s, std_dur_s, cv_dur.
    """
    n_cycles = len(cycles_df)
    mean_dur = float(cycles_df["duration_s"].mean()) if n_cycles > 0 else 0.0
    std_dur = float(cycles_df["duration_s"].std()) if n_cycles > 0 else 0.0
    cv_dur = std_dur / mean_dur if mean_dur > 0 else 1.0
    if n_cycles < GAIT_MIN_CYCLES_FAIL:
        flag = "fail"
    elif n_cycles < GAIT_MIN_CYCLES_WARN or cv_dur > GAIT_MAX_DURATION_CV:
        flag = "warn"
    else:
        flag = "pass"
    metrics = {
        "n_cycles": n_cycles,
        "mean_dur_s": round(mean_dur, 3),
        "std_dur_s": round(std_dur, 3),
        "cv_dur": round(cv_dur, 3),
    }
    return flag, metrics


def preprocessing_flag(n_brain_ics: int, n_ica_components: int) -> str:
    """Return the stage-2 flag from the ICA fit.

    Parameters
    ----------
    n_brain_ics : int
        Components kept by the IC rule.
    n_ica_components : int
        Components fitted.

    Returns
    -------
    str
        "pass", "warn" or "fail".
    """
    if n_brain_ics == 0 or n_ica_components < PREP_MIN_COMPONENTS:
        return "fail"
    return "warn" if n_brain_ics < MIN_BRAIN_ICS_WARN else "pass"


def ica_flag(n_brain_components: int, brain_fraction: float) -> str:
    """Return the stage-3 flag from the kept components.

    Parameters
    ----------
    n_brain_components : int
        Components kept.
    brain_fraction : float
        Kept / fitted components.

    Returns
    -------
    str
        "pass", "warn" or "fail".
    """
    if n_brain_components == 0:
        return "fail"
    if n_brain_components < MIN_BRAIN_ICS_WARN or brain_fraction < ICA_MIN_BRAIN_FRACTION:
        return "warn"
    return "pass"


def gait_epochs_flag(n_extracted: int, reject_fraction: float) -> str:
    """Return the stage-4 flag from the extracted gait-cycle segments.

    Parameters
    ----------
    n_extracted : int
        Cycles kept.
    reject_fraction : float
        Fraction of cycles rejected.

    Returns
    -------
    str
        "pass", "warn" or "fail".
    """
    if n_extracted < EPOCHS_MIN_FAIL:
        return "fail"
    if n_extracted < EPOCHS_MIN_WARN or reject_fraction > EPOCHS_MAX_REJECT_FRACTION:
        return "warn"
    return "pass"


def ersp_flag(n_nan: int, n_cycles_used: int, ersp_range_db: float, baseline_ok: bool) -> str:
    """Return the stage-5 flag; an implausible baseline always gives "warn".

    Parameters
    ----------
    n_nan : int
        NaN values in the ERSP.
    n_cycles_used : int
        Cycles averaged.
    ersp_range_db : float
        Max - min of the ERSP, dB.
    baseline_ok : bool
        Mean standing power within BASELINE_POWER_RANGE_V2.

    Returns
    -------
    str
        "pass", "warn" or "fail".
    """
    if not baseline_ok:
        return "warn"
    if n_nan > 0 or n_cycles_used < ERSP_MIN_CYCLES_FAIL:
        return "fail"
    if n_cycles_used < ERSP_MIN_CYCLES_WARN or ersp_range_db < ERSP_MIN_RANGE_DB:
        return "warn"
    return "pass"


class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars and arrays to Python types for JSON."""

    def default(self, obj):  # noqa: D102
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log_qc(qc_dir: Path, subject: str, stage: str, flag: str, metrics: dict) -> None:
    """Write the JSON QC record of one subject and stage.

    Parameters
    ----------
    qc_dir : pathlib.Path
        Folder of the per-subject QC records (created if absent).
    subject : str
        Subject ID, e.g. "S1".
    stage : str
        One of STAGE_ORDER.
    flag : str
        "pass", "warn" or "fail".
    metrics : dict
        Scalar values to record.
    """
    if flag not in ("pass", "warn", "fail"):
        raise ValueError(f"flag must be 'pass', 'warn', or 'fail', got '{flag}'")
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    record = {"subject": subject, "stage": stage, "flag": flag, **metrics}
    with open(qc_dir / f"sub-{subject}_{stage}_qc.json", "w") as f:
        json.dump(record, f, indent=2, cls=_NumpyEncoder)


def write_qc_summary(qc_dir: Path, out_path: Path) -> pd.DataFrame:
    """Collect all per-subject QC records into one TSV and print flag counts per stage.

    Parameters
    ----------
    qc_dir : pathlib.Path
        Folder of the per-subject QC records.
    out_path : pathlib.Path
        TSV to write.

    Returns
    -------
    pandas.DataFrame
        One row per subject and stage, sorted by stage then subject.
    """
    records = []
    for f in sorted(Path(qc_dir).glob("*_qc.json")):
        with open(f) as fh:
            records.append(json.load(fh))
    if not records:
        print("  No QC records found.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["stage_rank"] = df["stage"].apply(lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)
    df = df.sort_values(["stage_rank", "subject"]).drop(columns="stage_rank").reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  QC summary -> {out_path}\n  Rows: {len(df)}")
    for stage in STAGE_ORDER:
        counts = df.loc[df["stage"] == stage, "flag"].value_counts().to_dict()
        if counts:
            print(
                f"  {stage:>15}: pass={counts.get('pass', 0)}  "
                f"warn={counts.get('warn', 0)}  fail={counts.get('fail', 0)}"
            )
    return df
