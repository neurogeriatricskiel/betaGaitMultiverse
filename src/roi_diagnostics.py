"""Channel-level checks of the reference beta contrast: ROI weights, sensitivity, ICA, heel strikes.

Used by scripts/qc/qc_roi_channel_sensitivity.py, qc_ica_audit.py and
qc_heelstrike_locked.py. Read-only: works on the stored stage-2 to stage-5
files and never changes them. The linear ROI is linear in the channels, so a
subject's stored ROI contrast equals sum_i w_i * c_i, with c_i the channel's
13-30 Hz double-stance minus swing ERSP; every ROI variant below reweights
the same c_i.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from src.config import DIR_QC, SUBJECTS
from src.cycle_tfr import annotation_segment
from src.ersp import BETA_FMAX, BETA_FMIN
from src.qc import load_primary_cohort
from src.reference_ersp import FREQS

BETA = (FREQS >= BETA_FMIN) & (FREQS <= BETA_FMAX)
FRONTAL = re.compile(r"^(Fp|AF)")
LATERAL = re.compile(r"^(F7|F8|FT\d+|T\d+|TP\d+)$")
N_NARROW = 5
HS_WINDOW_S = (-0.2, 0.4)
HS_BAND_HZ = (1.0, 40.0)


class PhaseMaps(NamedTuple):
    """Stored double-stance and swing ERSP maps of a cohort (stage 5)."""

    subjects: list[str]
    ch_names: list[str]
    infos: list[mne.Info]
    ds: np.ndarray  # (n_subjects, n_channels, n_freqs), dB
    sw: np.ndarray
    weights: np.ndarray  # (n_subjects, n_channels)


def reference_cohort(dataset: str, dirs: dict) -> list[str]:
    """Return the reference cohort: treadmill subjects with a stage-5 ERSP; overground primary."""
    if dataset == "jacobsen":
        return load_primary_cohort(DIR_QC)
    return [s for s in SUBJECTS if (dirs["ersp"] / f"sub-{s}_ersp_double_stance.npy").exists()]


def load_phase_maps(cohort: list[str], dirs: dict) -> PhaseMaps:
    """Load the stored phase maps, ROI weights and channel info of every cohort subject.

    Parameters
    ----------
    cohort : list of str
        Subject IDs.
    dirs : dict
        Dataset folders (src.paths.get_dataset_dirs).

    Returns
    -------
    PhaseMaps
        Raises AssertionError if the channel order differs between subjects.
    """
    infos, ds, sw, w = [], [], [], []
    for s in cohort:
        raw = mne.io.read_raw_fif(
            dirs["clean"] / f"sub-{s}_desc-icaClean_concat_raw.fif", verbose=False
        )
        infos.append(raw.info)
        assert raw.ch_names == infos[0].ch_names, f"channel order differs for {s}"
        ds.append(np.load(dirs["ersp"] / f"sub-{s}_ersp_double_stance.npy"))
        sw.append(np.load(dirs["ersp"] / f"sub-{s}_ersp_swing.npy"))
        w.append(np.load(dirs["ersp"] / f"sub-{s}_roi_weights.npy"))
    return PhaseMaps(
        list(cohort), infos[0].ch_names, infos, np.stack(ds), np.stack(sw), np.stack(w)
    )


def channel_contrast(maps: PhaseMaps) -> np.ndarray:
    """Per subject and channel: mean 13-30 Hz double stance minus swing, dB (n_subjects, n_ch)."""
    return (maps.ds - maps.sw)[:, :, BETA].mean(-1)


def channel_mask(ch_names: list[str], pattern: re.Pattern) -> np.ndarray:
    """Boolean mask of the channels whose name matches ``pattern``."""
    return np.array([bool(pattern.match(c)) for c in ch_names])


def distances_to_cz(info: mne.Info) -> np.ndarray:
    """3D distance of every channel to Cz, mm (montage positions of ``info``)."""
    pos = np.array([ch["loc"][:3] for ch in info["chs"]]) * 1000
    return np.linalg.norm(pos - pos[info.ch_names.index("Cz")], axis=1)


def roi_variants(maps: PhaseMaps) -> dict[str, np.ndarray]:
    """Weight matrices (n_subjects, n_ch) of the stored ROI and the sensitivity ROIs.

    Returns
    -------
    dict of str to numpy.ndarray
        "stored", "without_frontal" (Fp*/AF* set to 0), "without_frontal_lateral"
        (also F7/F8, FT*, T*, TP*) and "narrow" (the N_NARROW channels nearest
        Cz, equal weights). Not yet renormalised (see `roi_test`).
    """
    frontal = channel_mask(maps.ch_names, FRONTAL)
    lateral = channel_mask(maps.ch_names, LATERAL)
    no_frontal = maps.weights * ~frontal
    narrow = np.zeros_like(maps.weights)
    for k, info in enumerate(maps.infos):
        narrow[k, np.argsort(distances_to_cz(info))[:N_NARROW]] = 1
    return {
        "stored": maps.weights.copy(),
        "without_frontal": no_frontal,
        "without_frontal_lateral": no_frontal * ~lateral,
        "narrow": narrow,
    }


def roi_test(weights: np.ndarray, contrast: np.ndarray) -> dict:
    """One-sample t-test of the renormalised ROI contrast against 0 (= paired DS vs swing).

    Parameters
    ----------
    weights : numpy.ndarray, shape (n_subjects, n_ch)
        ROI weights; each row is renormalised to sum 1.
    contrast : numpy.ndarray, shape (n_subjects, n_ch)
        Output of `channel_contrast`.

    Returns
    -------
    dict
        values (per subject), t, df, p, mean (dB), n_channels (non-zero weights).
    """
    w = weights / weights.sum(1, keepdims=True)
    values = (w * contrast).sum(1)
    t, p = ttest_1samp(values, 0)
    return {
        "values": values,
        "t": float(t),
        "df": len(values) - 1,
        "p": float(p),
        "mean": float(values.mean()),
        "n_channels": int((weights[0] > 0).sum()),
    }


def heel_strike_times(cycles: pd.DataFrame) -> np.ndarray:
    """Right and left heel strikes of a cycle table, seconds from walk onset."""
    return np.concatenate([cycles["rhs_start_s"].to_numpy(), cycles["lhs_s"].to_numpy()])


def heelstrike_average(
    raw: mne.io.BaseRaw, times_s: np.ndarray, picks: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Heel-strike-locked average of the 1-40 Hz walking signal, baseline -200..0 ms.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preloaded recording with a CS (walking) annotation.
    times_s : numpy.ndarray
        Heel-strike times, seconds from walk onset.
    picks : list of str
        Channels.

    Returns
    -------
    times_ms : numpy.ndarray
        Epoch time axis, ms.
    average : numpy.ndarray, shape (len(picks), n_times)
        Average in microvolts.
    """
    seg = annotation_segment(raw, "CS").copy().pick(picks).filter(*HS_BAND_HZ, verbose=False)
    sf = seg.info["sfreq"]
    tmin, tmax = HS_WINDOW_S
    idx = np.round(times_s * sf).astype(int)
    idx = idx[(idx + int(tmin * sf) >= 0) & (idx + int(round(tmax * sf)) < seg.n_times)]
    events = np.column_stack([idx + seg.first_samp, np.zeros_like(idx), np.ones_like(idx)])
    epochs = mne.Epochs(
        seg, events, tmin=tmin, tmax=tmax, baseline=(tmin, 0), preload=True,
        reject_by_annotation=False, verbose=False,
    )  # fmt: skip
    return epochs.times * 1000, epochs.average().get_data() * 1e6


def walking_beta(raw: mne.io.BaseRaw, channel: str) -> np.ndarray:
    """One channel of the walking (CS) segment, band-passed 13-30 Hz (MNE FIR defaults)."""
    seg = annotation_segment(raw, "CS").copy().pick([channel])
    return seg.filter(BETA_FMIN, BETA_FMAX, verbose=False).get_data()[0]


def retained_projection(
    raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA, include: list[int]
) -> mne.io.BaseRaw:
    """Back-projection of the ``include`` components only (no PCA residual)."""
    exclude = [j for j in range(ica.n_components_) if j not in include]
    return ica.apply(
        raw.copy(), include=include, exclude=exclude, n_pca_components=ica.n_components_,
        verbose=False,
    )  # fmt: skip
