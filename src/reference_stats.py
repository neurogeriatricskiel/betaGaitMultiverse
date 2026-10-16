"""Per-subject beta values and the double-stance vs swing test of the reference pipeline."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from src.ersp import beta_roi_scalar
from src.reference_ersp import FREQS


def phase_beta_values(subject: str, ersp_dir: Path) -> tuple[float, float] | None:
    """Beta (13-30 Hz) ROI value of one subject in double stance and in swing.

    Parameters
    ----------
    subject : str
        Subject ID.
    ersp_dir : pathlib.Path
        Folder of the prepana05 outputs.

    Returns
    -------
    tuple of float or None
        (double stance, swing) in dB; None if an input file is missing.
    """
    paths = [ersp_dir / f"sub-{subject}_{k}.npy" for k in ("ersp_double_stance", "ersp_swing")]
    weights_path = ersp_dir / f"sub-{subject}_roi_weights.npy"
    if not all(p.exists() for p in [*paths, weights_path]):
        return None
    weights = np.load(weights_path)
    beta_ds, beta_sw = (beta_roi_scalar(np.load(p), weights, FREQS) for p in paths)
    return beta_ds, beta_sw


def paired_test(beta_ds: Sequence[float], beta_sw: Sequence[float]) -> dict:
    """Paired t-test of double stance vs swing across subjects (two-sided).

    Parameters
    ----------
    beta_ds, beta_sw : array-like
        Per-subject values, dB.

    Returns
    -------
    dict
        n, df, t, p, mean_diff (dB) and n_ds_gt_swing.
    """
    beta_ds, beta_sw = np.asarray(beta_ds), np.asarray(beta_sw)
    t_stat, p_val = ttest_rel(beta_ds, beta_sw)
    diff = beta_ds - beta_sw
    return {
        "n": len(diff),
        "df": len(diff) - 1,
        "t": float(t_stat),
        "p": float(p_val),
        "mean_diff": float(diff.mean()),
        "n_ds_gt_swing": int(np.sum(diff > 0)),
    }
