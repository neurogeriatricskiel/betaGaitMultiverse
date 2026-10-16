"""ERSP under two averaging orders, recomputed per cycle (QC only; both datasets).

ours   : ERSP = mean_k 10*log10(P_k / P_standing)       (prepana05; the stored ERSP)
source : ERSP = 20*log10(mean_k |X_k| / mean |X_standing|), |X| = sqrt(P)
         (Jacobsen et al.'s order)

The TFR, edge crop, amplitude rejection, standing baseline and warp are
those of prepana05 (src.cycle_tfr, src.reference_ersp); the recomputed
"ours" is compared with the stored ERSP. Used by
scripts/qc/qc_averaging_order.py and qc_log_averaging_bias.py.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.config import AMP_THRESH, BASELINE_EPOCH_DUR_S, TFR_N_POINTS
from src.cycle_tfr import cycle_event_indices, cycle_power, standing_segment
from src.ersp import BETA_FMAX, BETA_FMIN, warp_cycle_to_grid
from src.plot_style import CMAP
from src.reference_ersp import FREQS, N_CYCLES

BETA_MASK = (FREQS >= BETA_FMIN) & (FREQS <= BETA_FMAX)
ORDERS = ("ours", "source")
ERSP_COHORTS = ("PRIMARY", "FULL")


def standing_power_and_magnitude(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray, list]:
    """Standing baseline as mean power and as mean magnitude (prepana05's epochs).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with one STAND annotation.

    Returns
    -------
    power, magnitude : numpy.ndarray, shape (n_channels, n_freqs)
        Mean over epochs and time of P and of sqrt(P).
    channels : list of str
        Channel names of the standing data.
    """
    rs = standing_segment(raw, "baseline")
    ev = mne.make_fixed_length_events(rs, duration=BASELINE_EPOCH_DUR_S)
    ep = mne.Epochs(
        rs,
        ev,
        tmin=0,
        tmax=BASELINE_EPOCH_DUR_S,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )
    ep.drop_bad(reject=dict(eeg=AMP_THRESH))
    bp = bm = 0
    nb = 0
    for e in ep.get_data():
        if np.abs(e).max() > AMP_THRESH:
            continue
        p, _ = cycle_power(e, rs.info["sfreq"], FREQS, N_CYCLES)
        bp = bp + p.mean(-1)
        bm = bm + np.sqrt(p).mean(-1)
        nb += 1
    return bp / nb, bm / nb, rs.ch_names


def cycle_ersp_orders(
    segs: Sequence[np.ndarray],
    ref_idx: list[int],
    sf: float,
    meta: pd.DataFrame,
    bp: np.ndarray,
    bm: np.ndarray,
    anchors: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Both-order ERSP from gait-cycle segments (prepana05's per-cycle loop).

    Parameters
    ----------
    segs : sequence of numpy.ndarray
        Cycle segments (n_channels, n_samples), RHS to RHS.
    ref_idx : list of int
        Channel rows to use, in the order of ``bp`` / ``bm``.
    sf : float
        Sampling rate, Hz.
    meta : pandas.DataFrame
        Matching cycle rows (rhs_start_s, lto_s, lhs_s, rto_s).
    bp, bm : numpy.ndarray, shape (n_channels, n_freqs)
        Standing power and magnitude.
    anchors : tuple of float
        Warp anchors, % of the cycle.

    Returns
    -------
    ours, source : numpy.ndarray, shape (n_channels, n_freqs, TFR_N_POINTS)
        ERSP (dB) under each order.
    n : int
        Cycles used.
    """
    s_log = s_mag = 0
    n = 0
    for i, seg in enumerate(segs):
        seg = seg[ref_idx]
        if np.abs(seg).max() > AMP_THRESH:
            continue
        p, c = cycle_power(seg, sf, FREQS, N_CYCLES)
        idx = cycle_event_indices(meta.iloc[i], sf, c)
        try:
            pw = warp_cycle_to_grid(p, *idx, anchors=anchors, n_points=TFR_N_POINTS)
            mg = warp_cycle_to_grid(np.sqrt(p), *idx, anchors=anchors, n_points=TFR_N_POINTS)
        except ValueError:
            continue
        s_log = s_log + 10 * np.log10(pw / bp[..., None])
        s_mag = s_mag + mg
        n += 1
    return s_log / n, 20 * np.log10((s_mag / n) / bm[..., None]), n


def subject_ersp_orders(
    sub: str, dirs: dict, anchors: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the stored ERSP, source-order ERSP and ROI weights of one subject.

    Parameters
    ----------
    sub : str
        Subject ID.
    dirs : dict
        Dataset folders (src.paths.get_dataset_dirs).
    anchors : tuple of float
        Reference warp anchors.

    Returns
    -------
    stored, source : numpy.ndarray, shape (n_channels, n_freqs, TFR_N_POINTS)
        ERSP in dB under our order (as stored) and the source order.
    weights : numpy.ndarray
        Linear Cz-ROI weights.
    """
    raw = mne.io.read_raw_fif(
        dirs["clean"] / f"sub-{sub}_desc-icaClean_concat_raw.fif", preload=True, verbose=False
    )
    bp, bm, stand_ch = standing_power_and_magnitude(raw)
    segs = np.load(dirs["gaitepochs"] / f"sub-{sub}_gait_segments.npy", allow_pickle=True)
    sf = float(np.load(dirs["gaitepochs"] / f"sub-{sub}_gait_sfreq.npy"))
    meta = pd.read_csv(dirs["gaitepochs"] / f"sub-{sub}_cycles_kept.tsv", sep="\t")
    ref_idx = [raw.ch_names.index(ch) for ch in stand_ch]
    ours, source, n = cycle_ersp_orders(segs, ref_idx, sf, meta, bp, bm, anchors)
    stored = np.load(dirs["ersp"] / f"sub-{sub}_ersp_beta.npy")
    w = np.load(dirs["ersp"] / f"sub-{sub}_roi_weights.npy")
    print(
        f"sub-{sub}: cycles={n}  max|recomputed-stored|={np.abs(ours - stored).max():.2e} dB",
        flush=True,
    )
    return stored, source, w


def order_contrast_lines(
    per_subject: dict,
    variants: dict,
    cohorts: Sequence[str],
    phase_idx: tuple[np.ndarray, np.ndarray],
    ersp_style: bool,
) -> list[str]:
    """Paired t-test of ROI beta double stance vs swing per cohort and averaging order.

    Parameters
    ----------
    per_subject : dict
        Subject -> (ours, source) ROI beta arrays (beta rows x cycle points).
    variants : dict
        Cohort name -> subject IDs.
    cohorts : sequence of str
        Cohorts to report, in order.
    phase_idx : tuple
        (double-stance indices, swing indices) on the cycle grid.
    ersp_style : bool
        Line format of jacobsen_ersp_averaging_order.txt (True) or of
        jacobsen_gpm_averaging_order.txt (False).

    Returns
    -------
    list of str
    """
    ds_idx, sw_idx = phase_idx
    lines = ["", "Beta 13-30 Hz double stance vs swing (ROI), paired t-test:"]
    for cohort in cohorts:
        subs = variants[cohort]
        for k, order in enumerate(ORDERS):
            ds = np.array([per_subject[s][k][:, ds_idx].mean() for s in subs])
            sw = np.array([per_subject[s][k][:, sw_idx].mean() for s in subs])
            t, p = ttest_rel(ds, sw)
            d = ds - sw
            counts = f"DS > swing {int((d > 0).sum())}/{len(subs)}"
            if ersp_style:
                lines.append(
                    f"  {cohort:<7s} {order:<6s}: t({len(subs) - 1}) = {t:.3f}  p = {p:.4g}  "
                    f"mean difference = {d.mean():+.3f} dB  df = {len(subs) - 1}  {counts}"
                )
            else:
                lines.append(
                    f"  {cohort:<7s} n={len(subs)}  {order:<6s}: t({len(subs) - 1}) = {t:.3f}  "
                    f"p = {p:.4g}  mean = {d.mean():+.3f} dB  {counts}"
                )
    return lines


def gpm_summary_lines(ours: np.ndarray, source: np.ndarray, n: int) -> list[str]:
    """Agreement of the group GPM under the two orders (correlations, amplitude ratio)."""
    beta = BETA_MASK
    slope = np.dot(ours.ravel(), source.ravel()) / np.dot(ours.ravel(), ours.ravel())
    return [
        f"n={n}  pixel corr(ours, source)={np.corrcoef(ours.ravel(), source.ravel())[0, 1]:.4f}",
        f"beta-trace corr={np.corrcoef(ours[beta].mean(0), source[beta].mean(0))[0, 1]:.4f}",
        f"99th pct |GPM| ours={np.percentile(np.abs(ours), 99):.3f}  "
        f"source={np.percentile(np.abs(source), 99):.3f}",
        f"source/ours amplitude ratio (lstsq slope)={slope:.3f}",
        f"max |source-ours| = {np.abs(source - ours).max():.3f} dB",
    ]


def ersp_map_lines(
    per_subject: dict, variants: dict, cohorts: Sequence[str] = ERSP_COHORTS
) -> list[str]:
    """Range, mean, % negative and maximum of the group ROI beta map per cohort and order."""
    lines = ["Group-mean ROI beta ERSP map (13-30 Hz rows x 101 cycle points), dB re standing:"]
    for cohort in cohorts:
        subs = variants[cohort]
        for k, order in enumerate(ORDERS):
            m = np.mean([per_subject[s][k] for s in subs], axis=0)
            max_pos = f"{m.max():+.3f} dB" if m.max() > 0 else "none (all pixels <= 0)"
            lines.append(
                f"  {cohort:<7s} n={len(subs):<2d} {order:<6s}: range {m.min():+.3f} to "
                f"{m.max():+.3f} dB  mean {m.mean():+.3f} dB  "
                f"negative {100 * (m < 0).mean():.1f} % of {m.size} pixels  max positive {max_pos}"
            )
    return lines


def sign_change_lines(
    per_subject: dict,
    variants: dict,
    cohorts: Sequence[str] = ERSP_COHORTS,
    listed: str = "FULL",
) -> list[str]:
    """Per-subject ROI-mean beta ERSP under both orders and the subjects whose sign flips.

    Subjects of cohort ``listed`` are printed one per line; the sign-change
    count is then given for each of ``cohorts``.
    """
    lines = ["", "Per-subject ROI-mean beta ERSP (beta rows x whole cycle), dB re standing:"]
    changed = []
    for s in variants[listed]:
        ours, source = (float(per_subject[s][k].mean()) for k in range(2))
        flip = np.sign(ours) != np.sign(source)
        changed += [s] if flip else []
        lines.append(
            f"  sub-{s}: ours {ours:+.3f}  source {source:+.3f}" + ("  SIGN CHANGE" if flip else "")
        )
    for cohort in cohorts:
        flipped = [s for s in changed if s in variants[cohort]]
        lines.append(
            f"  {cohort}: {len(flipped)} of {len(variants[cohort])} subjects change sign"
            + (f" ({', '.join(flipped)})" if flipped else "")
        )
    return lines


def save_gpm_order_plot(
    ours: np.ndarray, source: np.ndarray, anchors: tuple[float, float, float], out_path: Path
) -> None:
    """QC plot: group GPM under each order and their difference."""
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    v = float(np.percentile(np.abs(np.concatenate([ours, source])), 99))
    extent = [-0.5, 100.5, FREQS[0] - 0.5, FREQS[-1] + 0.5]
    for a_, arr, t in [
        (ax[0], ours, "ours: mean 10log10(P/B)"),
        (ax[1], source, "source: 20log10(mean|X|/mean|X_stand|)"),
    ]:
        im = a_.imshow(
            arr, aspect="auto", origin="lower", cmap=CMAP, vmin=-v, vmax=v, extent=extent
        )
        a_.set_title(f"GPM, {t}")
        for x in anchors:
            a_.axvline(x, color="k", ls=":")
    fig.colorbar(im, ax=ax[:2])
    diff = source - ours
    vd = float(np.percentile(np.abs(diff), 99))
    im = ax[2].imshow(
        diff, aspect="auto", origin="lower", cmap=CMAP, vmin=-vd, vmax=vd, extent=extent
    )
    ax[2].set_title("source - ours")
    fig.colorbar(im, ax=ax[2])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
