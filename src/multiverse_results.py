"""Read-only access to the combined multiverse results and their per-universe statistics.

Results come from COMET's comet/results/multiverse_results.pkl. Measure per
dataset (per-subject double-stance - swing beta contrast, dB):
  stepup   -- "t_stats" (historical field name; the values are dB
              differences, not t), MULTIVERSE_SUBJECTS (n = 15).
  jacobsen -- "dedup_diffs": the same contrasts on the primary
              independent-recordings cohort (n = 14).
Per universe: mean, one-sample t-test against 0 (two-sided, uncorrected)
and a t-based CI -- the computation comet's specification_curve() used.
"""

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def measure_for(dataset: str) -> str:
    """Field of the pkl holding the per-subject contrasts reported for this dataset."""
    return "dedup_diffs" if dataset == "jacobsen" else "t_stats"


def load_combined_results(comet_dir: Path | str) -> dict:
    """{"universe_N": result dict} from comet/results/multiverse_results.pkl."""
    with open(Path(comet_dir) / "results" / "multiverse_results.pkl", "rb") as f:
        return pickle.load(f)


def decisions_dict(raw_decisions: dict) -> dict:
    """{'Decision 1': 'highpass_hz', 'Value 1': 0.5, ...} -> {'highpass_hz': 0.5, ...}."""
    out, i = {}, 1
    while f"Decision {i}" in raw_decisions:
        out[raw_decisions[f"Decision {i}"]] = raw_decisions[f"Value {i}"]
        i += 1
    return out


def contrast_statistics(values: Sequence[float], ci_level: int = 95) -> dict:
    """Mean, one-sample t-test against 0 and t-based CI of per-subject contrasts.

    Parameters
    ----------
    values : array-like
        Per-subject contrasts, dB (non-finite values are dropped).
    ci_level : int
        Confidence level, %.

    Returns
    -------
    dict
        n, mean, t, p, ci_lo, ci_hi.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    mean = float(vals.mean())
    t_stat, p_val = stats.ttest_1samp(vals, 0.0)
    half = (
        vals.std(ddof=1) / np.sqrt(vals.size) * stats.t.ppf((1 + ci_level / 100) / 2, vals.size - 1)
    )
    return {
        "n": vals.size,
        "mean": mean,
        "t": float(t_stat),
        "p": float(p_val),
        "ci_lo": mean - half,
        "ci_hi": mean + half,
    }


def universe_table(all_universes: dict, measure: str, ci_level: int = 95) -> pd.DataFrame:
    """One row per universe: decisions, n, mean (dB), t, p, CI bounds; sorted by mean.

    Parameters
    ----------
    all_universes : dict
        Output of ``load_combined_results``.
    measure : str
        Per-subject contrast field ("t_stats" or "dedup_diffs").
    ci_level : int
        Confidence level, %.

    Returns
    -------
    pandas.DataFrame
    """
    records = [
        {
            "universe": int(key.split("_")[1]),
            **decisions_dict(data["__decisions"]),
            **contrast_statistics(data[measure], ci_level),
        }
        for key, data in all_universes.items()
    ]
    return pd.DataFrame(records).sort_values("mean", kind="stable").reset_index(drop=True)
