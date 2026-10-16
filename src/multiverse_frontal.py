"""Frontal diagnostics per multiverse universe (scripts/qc/qc_multiverse_frontal.py). Read-only.

For one subject in one universe, from that universe's cached ICA-corrected
data (results/multiverse/<ds>/branches/), the channel-level 13-30 Hz
double-stance and swing ERSP is computed exactly as
src.multiverse_pipeline.cycle_ersp_from_clean_raw, restricted to the beta
rows (Morlet rows and the edge crop do not depend on the other frequencies,
so the rows are identical to the full computation), together with the
heel-strike-locked 1-40 Hz average at Fpz and Cz. The caches are float32
files, whereas the stored universe results were computed from float64 data
in memory, so subject values differ slightly from the stored ones.
"""

from __future__ import annotations

import os

import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src import multiverse_pipeline as mp
from src import roi_diagnostics as rd
from src.config import AMP_THRESH, MIN_GAIT_CYCLES, ROI_CENTER_CH, TFR_EDGE_CROP, TFR_N_POINTS
from src.cycle_tfr import (
    annotation_segment,
    cycle_event_indices,
    cycle_power,
    log_ratio_per_cycle,
    standing_segment,
)
from src.ersp import (
    BETA_FMAX,
    BETA_FMIN,
    load_group_anchors,
    phase_split_indices,
    warp_cycle_to_grid,
)
from src.paths import get_dataset_dirs
from src.spatial_filter import linear_roi_weights

BETA_ROWS = (mp.FREQS >= BETA_FMIN) & (mp.FREQS <= BETA_FMAX)
HS_CHANNELS = ["Fpz", "Cz"]


def beta_phase_maps(raw: mne.io.BaseRaw, subject: str, cycles: pd.DataFrame, anchors) -> tuple:
    """Double-stance and swing beta ERSP per channel, (n_ch, n_beta_freqs) each, and cycles used."""
    freqs, n_cycles = mp.FREQS[BETA_ROWS], mp.N_CYCLES_WAV[BETA_ROWS]
    base = mp.compute_standing_baseline(
        standing_segment(raw, subject), freqs, n_cycles, edge_crop=TFR_EDGE_CROP,
        amp_thresh=AMP_THRESH,
    )  # fmt: skip
    walk = annotation_segment(raw, "CS")
    sf, data, warped = walk.info["sfreq"], walk.get_data(), []
    for _, row in cycles.iterrows():
        i0, i1 = int(round(row["rhs_start_s"] * sf)), int(round(row["rhs_end_s"] * sf))
        if i0 < 0 or i1 > data.shape[1] or i1 <= i0:
            continue
        seg = data[:, i0:i1]
        if not np.isfinite(seg).all() or np.max(np.abs(seg)) > AMP_THRESH:
            continue
        power, crop = cycle_power(seg, sf, freqs, n_cycles)
        lto, lhs, rto = cycle_event_indices(row, sf, crop)
        try:
            warped.append(
                warp_cycle_to_grid(power, lto, lhs, rto, anchors=anchors, n_points=TFR_N_POINTS)
            )
        except ValueError:
            continue
    if len(warped) < MIN_GAIT_CYCLES:
        raise RuntimeError(f"sub-{subject}: only {len(warped)} gait cycles accepted")
    ersp = log_ratio_per_cycle(warped, base)
    ds_idx, sw_idx = phase_split_indices(anchors, TFR_N_POINTS)
    return ersp[..., ds_idx].mean(axis=(0, -1)), ersp[..., sw_idx].mean(axis=(0, -1)), len(warped)


def subject_universe(job: tuple[str, dict, str]) -> dict:
    """One subject in one universe: beta phase maps, ROI weights, positions, heel-strike average.

    Parameters
    ----------
    job : tuple
        (dataset, universe decisions incl. lowpass_hz, subject ID). The
        dataset is set in the worker's environment before src.config is read.

    Returns
    -------
    dict
        ds, sw, weights, ch_names, pos, hs_avg (uV, Fpz and Cz), hs_times_ms, n_cycles.
    """
    dataset, decisions, subject = job
    assert os.environ.get("BETAGAIT_DATASET") == dataset, "worker dataset mismatch"
    dirs = get_dataset_dirs(dataset)
    rule = decisions["iclabel_rule"]
    path = mp.branch_dir_for(subject, decisions) / f"sub-{subject}_desc-icaClean_{rule}_raw.fif"
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    cycles = pd.read_csv(dirs["gait_events"] / f"sub-{subject}_cycles.tsv", sep="\t")
    ds, sw, n = beta_phase_maps(raw, subject, cycles, load_group_anchors(dirs["ersp"]))
    times, avg = rd.heelstrike_average(raw, rd.heel_strike_times(cycles), HS_CHANNELS)
    return {
        "ds": ds, "sw": sw, "weights": linear_roi_weights(raw.info, center_ch=ROI_CENTER_CH),
        "ch_names": list(raw.ch_names), "pos": np.array([c["loc"][:3] for c in raw.info["chs"]]),
        "hs_avg": avg, "hs_times_ms": times, "n_cycles": n,
    }  # fmt: skip


def universe_row(results: list[dict]) -> dict:
    """Heel-strike peak-to-peak and the three ROI tests of one universe's subjects."""
    names = results[0]["ch_names"]
    assert all(r["ch_names"] == names for r in results), "channel order differs"
    weights = np.stack([r["weights"] for r in results])
    contrast = np.stack([(r["ds"] - r["sw"]).mean(-1) for r in results])
    narrow = np.zeros_like(weights)
    for k, r in enumerate(results):
        d = np.linalg.norm(r["pos"] - r["pos"][names.index("Cz")], axis=1)
        narrow[k, np.argsort(d)[: rd.N_NARROW]] = 1
    frontal = rd.channel_mask(names, rd.FRONTAL)
    variants = {"roi": weights, "nofrontal": weights * ~frontal, "narrow": narrow}
    hs = np.mean([r["hs_avg"] for r in results], axis=0)
    after = results[0]["hs_times_ms"] >= 0
    p2p = hs[:, after].max(1) - hs[:, after].min(1)
    row = {"n": len(results), "fpz_p2p_uv": p2p[0], "cz_p2p_uv": p2p[1]}
    tests = {k: rd.roi_test(w, contrast) for k, w in variants.items()}
    for k, r in tests.items():
        row |= {f"{k}_db": r["mean"], f"{k}_t": r["t"], f"{k}_p": r["p"]}
    row["roi_values"] = tests["roi"]["values"]
    return row


def correlation_lines(table: pd.DataFrame) -> list[str]:
    """Return Pearson and Spearman of each deflection with the stored contrast (all, per ASR)."""
    lines = []
    groups = [("all universes", table)] + [(f"ASR {a}", g) for a, g in table.groupby("asr_mode")]
    for channel in ("fpz", "cz"):
        for label, g in groups:
            x, y = g[f"{channel}_p2p_uv"], g["stored_roi_db"]
            r, p_r = pearsonr(x, y)
            rho, p_rho = spearmanr(x, y)
            lines.append(
                f"  {channel.capitalize():<3} {label:<14} k = {len(g):>2}  Pearson r = {r:+.2f} "
                f"(p = {p_r:.2g})  Spearman rho = {rho:+.2f} (p = {p_rho:.2g})"
            )
    return lines
