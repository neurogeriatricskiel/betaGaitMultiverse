"""Single-subject multiverse branch (mulana02) and the universe ERSP of mulana04.

Three decision nodes (highpass_hz, asr_mode, iclabel_rule). The same
event-anchored double-stance - swing beta contrast as the reference
pipeline, reusing src.cycle_tfr and src.ersp; TFR range 8-40 Hz
(MULTIVERSE_TFR_FMAX). ICA is fitted once per (highpass_hz, asr_mode,
lowpass_hz) branch and cached; the three IC rules reuse it.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject

from src.config import (
    AMP_THRESH,
    ASR_CUTOFF_BY_MODE,
    AUTOREJECT_N_INTERPOLATE,
    DATASET,
    DIR_MULTIVERSE_BRANCHES,
    EPOCH_DUR,
    ICA_FIT_PARAMS,
    ICA_METHOD,
    MIN_CLEAN_EPOCHS,
    MIN_GAIT_CYCLES,
    MULTIVERSE_TFR_FMAX,
    N_COMPONENTS,
    RANDOM_STATE,
    ROI_CENTER_CH,
    TFR_EDGE_CROP,
    TFR_FMIN,
    TFR_N_CYCLES_DIVISOR,
    TFR_N_POINTS,
)
from src.cycle_tfr import (
    annotation_segment,
    cycle_event_indices,
    cycle_power,
    log_ratio_per_cycle,
    standing_segment,
)
from src.ersp import (
    beta_roi_scalar,
    compute_standing_baseline,
    load_group_anchors,
    phase_split_indices,
    warp_cycle_to_grid,
)
from src.ica_utils import iclabel_probabilities, run_ica, select_ics_by_rule
from src.paths import get_dataset_dirs
from src.pipeline_steps import apply_ica, load_and_concatenate, preprocess_raw
from src.preprocessing import drop_invalid_eeg_channels
from src.spatial_filter import apply_linear_roi, linear_roi_weights

FREQS = np.arange(TFR_FMIN, int(MULTIVERSE_TFR_FMAX) + 1, dtype=float)  # 8-40 Hz
N_CYCLES_WAV = FREQS / TFR_N_CYCLES_DIVISOR


def branch_dir_for(subject: str, decisions: dict) -> Path:
    """Return the branch-cache folder of one subject and universe.

    Keyed on the ICA-relevant decisions only (highpass_hz, asr_mode,
    lowpass_hz). iclabel_rule is deliberately excluded: all
    three ICLabel rules are applied on top of the SAME cached ICA fit
    and ICLabel run, so this key gives 3 x 3 = 9 unique branches per
    subject instead of 27.
    """
    ica_keys = {"highpass_hz", "asr_mode", "lowpass_hz"}
    parts = "_".join(f"{k}-{decisions[k]}" for k in sorted(ica_keys) if k in decisions)
    d = DIR_MULTIVERSE_BRANCHES / subject / parts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fit_or_load_ica(raw, subject, branch_dir):
    """Fit ICA and ICLabel once per branch, or load the cached fit and probabilities.

    One fit per (highpass_hz, asr_mode, lowpass_hz) branch.

    Duplicates the AutoReject-epoching step from
    src.pipeline_steps.fit_ica() rather than calling it directly: that
    function bundles fitting with applying a single exclusion set and
    caches 1:1 on iclean_path, whereas here one ICA fit + ICLabel run
    must be reused across three different exclusion rules (see
    branch_dir_for). The actual exclusion + apply step below reuses
    src.pipeline_steps.apply_ica() unchanged.

    Returns
    -------
    ica   : fitted mne.preprocessing.ICA (exclude not yet set)
    probs : ndarray, shape (n_components, 7) -- ICLabel class probabilities
    """
    ica_path = branch_dir / f"sub-{subject}_ica.fif"
    probs_path = branch_dir / f"sub-{subject}_iclabel_probs.npy"

    if ica_path.exists() and probs_path.exists():
        ica = mne.preprocessing.read_ica(ica_path, verbose=False)
        probs = np.load(probs_path)
        return ica, probs

    epochs_raw = mne.make_fixed_length_epochs(raw, duration=EPOCH_DUR, preload=True, verbose=False)
    epochs_raw.pick("eeg")
    drop_invalid_eeg_channels(epochs_raw)

    ar = AutoReject(
        n_interpolate=AUTOREJECT_N_INTERPOLATE, random_state=RANDOM_STATE, verbose=False
    )
    ar.fit(epochs_raw)
    epochs_clean, _ = ar.transform(epochs_raw, return_log=True)

    if len(epochs_clean) < MIN_CLEAN_EPOCHS:
        raise RuntimeError(f"sub-{subject}: only {len(epochs_clean)} clean epochs")

    ica = run_ica(
        epochs_clean,
        n_components=N_COMPONENTS,
        method=ICA_METHOD,
        fit_params=ICA_FIT_PARAMS,
        random_state=RANDOM_STATE,
    )
    probs, _ = iclabel_probabilities(ica, epochs_clean)

    ica.save(ica_path, overwrite=True, verbose=False)
    np.save(probs_path, probs)
    return ica, probs


def run_subject_multiverse(subject: str, decisions: dict) -> dict:
    """Run one universe for one subject: preprocess, ICA (cached), ERSP, beta contrast.

    Parameters
    ----------
    subject : str
        Subject ID.
    decisions : dict
        "highpass_hz" (float), "asr_mode" ("off" / "sd3" / "sd20"),
        "iclabel_rule" ("conservative" / "balanced" / "liberal") and
        "lowpass_hz" (fixed by the universe template to
        src.config.MULTIVERSE_LOWPASS_HZ = 60 Hz; not a decision node).

    Returns
    -------
    dict
        Beta double stance, swing and their difference (dB), cycles used,
        brain ICs kept, and the decisions. Raises on failure; the universe
        scripts catch the error and skip the subject.
    """
    dirs = get_dataset_dirs(DATASET)
    raw_dir = dirs["raw"]
    event_dir = dirs["gait_events"]
    ersp_dir = dirs["ersp"]

    branch_dir = branch_dir_for(subject, decisions)
    iclabel_rule = decisions["iclabel_rule"]
    iclean_path = branch_dir / f"sub-{subject}_desc-icaClean_{iclabel_rule}_raw.fif"

    # Preprocessing -- identical call to the reference pipeline
    asr_mode = decisions["asr_mode"]
    use_asr = asr_mode != "off"
    asr_cutoff = ASR_CUTOFF_BY_MODE.get(asr_mode)  # None when asr_mode == "off" (unused)

    raw = load_and_concatenate(subject, raw_dir)
    raw = preprocess_raw(
        raw,
        subject,
        highpass_hz=float(decisions["highpass_hz"]),
        lowpass_hz=float(decisions["lowpass_hz"]),
        use_asr=use_asr,
        asr_cutoff=asr_cutoff,
        use_gedai=False,
    )

    # ICA: fit once per (highpass_hz, asr_mode, lowpass_hz) branch, cached;
    # apply this universe's iclabel_rule on top without refitting.
    ica, probs = _fit_or_load_ica(raw, subject, branch_dir)
    exclude_ics = select_ics_by_rule(probs, iclabel_rule)
    ica.exclude = exclude_ics
    n_brain = ica.n_components_ - len(exclude_ics)

    raw_clean = apply_ica(raw, ica, subject, iclean_path=iclean_path)

    anchors = load_group_anchors(ersp_dir)  # shared with the reference pipeline
    cycles = pd.read_csv(event_dir / f"sub-{subject}_cycles.tsv", sep="	")
    ersp_per_cycle = cycle_ersp_from_clean_raw(raw_clean, subject, cycles, anchors)
    n_cycles = ersp_per_cycle.shape[0]

    # Event-anchored double-stance / swing phase split (same anchors used
    # for warping above, so this is a fixed grid split -- see src.ersp)
    double_stance_idx, swing_idx = phase_split_indices(anchors, TFR_N_POINTS)

    ersp_double_stance = ersp_per_cycle[:, :, :, double_stance_idx].mean(
        axis=(0, -1)
    )  # (n_ch, n_freqs)
    ersp_swing = ersp_per_cycle[:, :, :, swing_idx].mean(axis=(0, -1))  # (n_ch, n_freqs)

    # Linear Cz-ROI beta-band reduction -- same as prepana07
    weights = linear_roi_weights(raw_clean.info, center_ch=ROI_CENTER_CH)
    beta_double_stance = beta_roi_scalar(ersp_double_stance, weights, FREQS)
    beta_swing = beta_roi_scalar(ersp_swing, weights, FREQS)

    print(
        f"  sub-{subject}: double_stance={beta_double_stance:+.2f} dB  "
        f"swing={beta_swing:+.2f} dB  diff={beta_double_stance - beta_swing:+.2f} dB  "
        f"n_cycles={n_cycles}  n_brain_ics={n_brain}"
    )

    return {
        "subject": subject,
        "beta_double_stance": float(beta_double_stance),
        "beta_swing": float(beta_swing),
        "beta_diff": float(beta_double_stance - beta_swing),
        "n_cycles": n_cycles,
        "n_brain_ics": int(n_brain),
        **decisions,
    }


def cycle_ersp_from_clean_raw(
    raw_clean: mne.io.BaseRaw,
    subject: str,
    cycles: pd.DataFrame,
    anchors: tuple[float, float, float],
) -> np.ndarray:
    """Per-cycle ERSP (dB re standing) of one subject from ICA-corrected data, 8-40 Hz.

    Parameters
    ----------
    raw_clean : mne.io.BaseRaw
        ICA-corrected STAND + CS recording.
    subject : str
        Subject ID, for error messages.
    cycles : pandas.DataFrame
        Gait cycles (rhs_start_s, rhs_end_s, lto_s, lhs_s, rto_s), seconds from walk onset.
    anchors : tuple of float
        Group-median (LTO, LHS, RTO) anchors, % of the cycle.

    Returns
    -------
    numpy.ndarray, shape (n_cycles, n_channels, n_freqs, TFR_N_POINTS)
    """
    raw_stand = standing_segment(raw_clean, subject)
    raw_walk = annotation_segment(raw_clean, "CS")
    sfreq = raw_walk.info["sfreq"]
    baseline_power = compute_standing_baseline(
        raw_stand, FREQS, N_CYCLES_WAV, edge_crop=TFR_EDGE_CROP, amp_thresh=AMP_THRESH
    )
    walk_data = raw_walk.get_data()
    tfr_cycles = []
    for _, row in cycles.iterrows():
        i0 = int(round(row["rhs_start_s"] * sfreq))
        i1 = int(round(row["rhs_end_s"] * sfreq))
        if i0 < 0 or i1 > walk_data.shape[1] or i1 <= i0:
            continue
        seg = walk_data[:, i0:i1]
        if not np.isfinite(seg).all() or np.max(np.abs(seg)) > AMP_THRESH:
            continue
        power, crop = cycle_power(seg, sfreq, FREQS, N_CYCLES_WAV)
        lto, lhs, rto = cycle_event_indices(row, sfreq, crop)
        try:
            tfr_cycles.append(
                warp_cycle_to_grid(power, lto, lhs, rto, anchors=anchors, n_points=TFR_N_POINTS)
            )
        except ValueError:
            continue
    if len(tfr_cycles) < MIN_GAIT_CYCLES:
        raise RuntimeError(f"sub-{subject}: only {len(tfr_cycles)} gait cycles accepted")
    return log_ratio_per_cycle(tfr_cycles, baseline_power)


def cached_universe_roi_ersp(
    subject: str, decisions: dict, gait_events_dir: Path, anchors: tuple[float, float, float]
) -> np.ndarray:
    """Group-ROI ERSP of one subject in one universe, from the cached ICA-corrected data.

    Parameters
    ----------
    subject : str
        Subject ID.
    decisions : dict
        highpass_hz, asr_mode, iclabel_rule, lowpass_hz of the universe.
    gait_events_dir : pathlib.Path
        Folder of the stage-1 cycle tables.
    anchors : tuple of float
        Group-median warp anchors.

    Returns
    -------
    numpy.ndarray, shape (n_freqs, TFR_N_POINTS)
        Mean over cycles, reduced with the linear Cz ROI.
    """
    rule = decisions["iclabel_rule"]
    iclean_path = branch_dir_for(subject, decisions) / f"sub-{subject}_desc-icaClean_{rule}_raw.fif"
    if not iclean_path.exists():
        raise FileNotFoundError(
            f"sub-{subject}: no cached ICA-corrected data at {iclean_path} -- "
            "run mulana02_run_multiverse.py for this universe first."
        )
    raw_clean = mne.io.read_raw_fif(iclean_path, preload=True, verbose=False)
    cycles = pd.read_csv(gait_events_dir / f"sub-{subject}_cycles.tsv", sep="	")
    ersp_mean = cycle_ersp_from_clean_raw(raw_clean, subject, cycles, anchors).mean(axis=0)
    weights = linear_roi_weights(raw_clean.info, center_ch=ROI_CENTER_CH)
    return apply_linear_roi(ersp_mean, weights)
