"""
qc_join_transient.py -- filter transient at the treadmill STAND/CS join and its effect (treadmill).

Purpose
    The treadmill STAND and CS recordings are separate files, joined with
    mne.concatenate_raws (src.pipeline_steps.load_and_concatenate) before
    the reference filters run, so the high-pass spans the join. Per
    reference subject:
      - the largest per-second variance of the Cz-ROI signal within +/-3 s
        of the join (stored d02_prep concat_raw), as a ratio to the median of
        the surrounding minute (20-60 s either side);
      - gait cycles starting within half the high-pass FIR length of walk
        onset, and whether they passed the AMP_THRESH rule;
      - the Cz-ROI beta contrast (double stance - swing) with every cycle
        starting within EARLY_S of walk onset removed, next to the stored
        contrast; group paired t-test both ways;
      - the bad-channel set (|z| of peak-to-peak > BAD_CHANNEL_ZSCORE) with
        +/-MASK_S around the join masked, next to the unmasked set;
      - whether the 2 s epoch containing the join was kept in the AutoReject
        epochs used for the ICA fit (drop log of preica_clean_epo.fif).
    Multiverse: for the two cached subjects with the largest join ratio, the
    same early-cycle removal in each 0.5 Hz, 60 Hz low-pass universe, from
    the float32 branch caches.
Inputs
    datasets/stepup/d00_raw/ (bad-channel recomputation)
    datasets/stepup/d02_prep/sub-*_concat_raw.fif, sub-*_preica_clean_epo.fif
    datasets/stepup/d03_clean/sub-*_desc-icaClean_concat_raw.fif
    datasets/stepup/d04_gaitepochs/, d05_ersp/ (stored ERSP, ROI weights, frozen anchors)
    datasets/stepup/d01_gaitevents/sub-*_cycles.tsv
    results/multiverse/stepup/branches/<subject>/<branch>/sub-*_desc-icaClean_<rule>_raw.fif
Outputs
    results/pipeline/stepup/qc/stepup_join_transient.txt, stepup_join_transient.tsv
Manuscript
    Methods 2.3 (preprocessing; concatenation before filtering).
Run
    python scripts/qc/qc_join_transient.py   (about 30 min, about 3 GB RAM)
"""

import os

os.environ["BETAGAIT_DATASET"] = "stepup"

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ttest_1samp  # noqa: E402

from src import config  # noqa: E402
from src import multiverse_pipeline as mp  # noqa: E402
from src.averaging_order import (  # noqa: E402
    BETA_MASK,
    cycle_ersp_orders,
    standing_power_and_magnitude,
)
from src.ersp import load_group_anchors, load_reference_anchors, phase_split_indices  # noqa: E402
from src.multiverse_frontal import beta_phase_maps  # noqa: E402
from src.paths import get_dataset_dirs  # noqa: E402
from src.pipeline_steps import load_and_concatenate  # noqa: E402
from src.roi_diagnostics import reference_cohort  # noqa: E402
from src.spatial_filter import apply_linear_roi, linear_roi_weights  # noqa: E402

EARLY_S = 20.0  # s -- cycles starting this soon after walk onset are removed
MASK_S = 5.0  # s -- masked either side of the join for the bad-channel z-score
NEAR_S = 3  # s -- window either side of the join for the largest variance ratio
REF_S = (20, 60)  # s -- surrounding-minute reference window, either side
RATIO_FLAG = 3.0  # ratio counted as a transient second
MULTIVERSE_HIGHPASS_HZ = 0.5
N_MULTIVERSE_SUBJECTS = 2

mne.set_log_level("ERROR")


def join_seconds(raw: mne.io.BaseRaw) -> float:
    """Return the join (CS onset) in seconds from the start of the recording."""
    cs = [a for a in raw.annotations if a["description"] == "CS"][0]
    return float(cs["onset"] - raw.first_time)


def highpass_fir_s(sfreq: float) -> float:
    """Length (s) of the reference FIR high-pass at this sampling rate."""
    h = mne.filter.create_filter(
        None, sfreq, config.REFERENCE_HIGHPASS_HZ, None, fir_design="firwin", verbose=False
    )
    return len(h) / sfreq


def variance_ratios(prep: mne.io.BaseRaw, join: float) -> dict[int, float]:
    """Per-second Cz-ROI variance within +/-20 s of the join over the surrounding-minute median."""
    sf = prep.info["sfreq"]
    roi = linear_roi_weights(prep.info, center_ch=config.ROI_CENTER_CH) @ prep.get_data(picks="eeg")

    def var(k):
        i = int((join + k) * sf)
        return float(roi[i : i + int(sf)].var())

    ref = [var(-k) for k in range(REF_S[0] + 1, REF_S[1] + 1)] + [var(k) for k in range(*REF_S)]
    med = float(np.median(ref))
    return {k: var(k) / med for k in range(-20, 20)}


def bad_channels(subject: str, dirs: dict, join: float) -> tuple[list[str], list[str]]:
    """Bad channels unmasked and with +/-MASK_S around the join masked.

    The filtering repeats src.pipeline_steps.preprocess_raw up to its
    bad-channel step (resample, high-pass, low-pass, notch).
    """
    raw = load_and_concatenate(subject, dirs["raw"])
    if raw.info["sfreq"] > config.TARGET_SFREQ:
        raw.resample(config.TARGET_SFREQ, verbose=False)
    raw.filter(l_freq=config.REFERENCE_HIGHPASS_HZ, h_freq=None, fir_design="firwin")
    raw.filter(l_freq=None, h_freq=config.REFERENCE_LOWPASS_HZ, fir_design="firwin")
    raw.notch_filter(freqs=config.LINE_FREQ)
    data = raw.get_data()
    keep = np.abs(raw.times - join) > MASK_S

    def bads(x):
        ptp = np.ptp(x, axis=1)
        z = (ptp - ptp.mean()) / (ptp.std() + 1e-12)
        return [raw.ch_names[i] for i in np.where(np.abs(z) > config.BAD_CHANNEL_ZSCORE)[0]]

    return bads(data), bads(data[:, keep])


def join_epoch_kept(subject: str, dirs: dict, join: float) -> tuple[int, str]:
    """Index of the 2 s ICA-fit epoch containing the join and its drop-log entry ('' = kept)."""
    epochs = mne.read_epochs(dirs["prep"] / f"sub-{subject}_preica_clean_epo.fif")
    k = int(join // config.EPOCH_DUR)
    return k, ",".join(epochs.drop_log[k])


def roi_contrast(ersp: np.ndarray, weights: np.ndarray, phase_idx) -> float:
    """Cz-ROI beta double stance - swing (dB) of one ERSP."""
    beta = apply_linear_roi(ersp, weights)[BETA_MASK]
    return float(beta[:, phase_idx[0]].mean() - beta[:, phase_idx[1]].mean())


def reference_contrasts(subject: str, dirs: dict, anchors, phase_idx) -> dict:
    """Return the stored contrast, its recomputation and the contrast without early cycles."""
    clean = mne.io.read_raw_fif(
        dirs["clean"] / f"sub-{subject}_desc-icaClean_concat_raw.fif", preload=True
    )
    bp, bm, ch = standing_power_and_magnitude(clean)
    idx = [clean.ch_names.index(c) for c in ch]
    segs = np.load(dirs["gaitepochs"] / f"sub-{subject}_gait_segments.npy", allow_pickle=True)
    sf = float(np.load(dirs["gaitepochs"] / f"sub-{subject}_gait_sfreq.npy"))
    meta = pd.read_csv(dirs["gaitepochs"] / f"sub-{subject}_cycles_kept.tsv", sep="\t")
    w = np.load(dirs["ersp"] / f"sub-{subject}_roi_weights.npy")
    stored = np.load(dirs["ersp"] / f"sub-{subject}_ersp_beta.npy")
    full, _, n_full = cycle_ersp_orders(segs, idx, sf, meta, bp, bm, anchors)
    late = (meta["rhs_start_s"] >= EARLY_S).to_numpy()
    cut, _, n_cut = cycle_ersp_orders(
        [s for s, k in zip(segs, late, strict=True) if k], idx, sf,
        meta[late].reset_index(drop=True), bp, bm, anchors,
    )  # fmt: skip
    return {
        "stored_db": roi_contrast(stored, w, phase_idx),
        "recomputed_db": roi_contrast(full, w, phase_idx),
        "without_early_db": roi_contrast(cut, w, phase_idx),
        "n_cycles": n_full,
        "n_cycles_without_early": n_cut,
    }


def subject_row(subject: str, dirs: dict, anchors, phase_idx) -> dict:
    """All join measures for one reference subject."""
    prep = mne.io.read_raw_fif(dirs["prep"] / f"sub-{subject}_concat_raw.fif", preload=True)
    join, fir = join_seconds(prep), highpass_fir_s(prep.info["sfreq"])
    ratios = variance_ratios(prep, join)
    del prep
    near = {k: r for k, r in ratios.items() if -NEAR_S <= k < NEAR_S}
    worst = max(near, key=near.get)
    cycles = pd.read_csv(dirs["gait_events"] / f"sub-{subject}_cycles.tsv", sep="\t")
    kept = pd.read_csv(dirs["gaitepochs"] / f"sub-{subject}_cycles_kept.tsv", sep="\t")
    early = cycles[cycles["rhs_start_s"] < fir / 2]
    bads, bads_masked = bad_channels(subject, dirs, join)
    epoch, drop = join_epoch_kept(subject, dirs, join)
    return {
        "subject": subject,
        "join_s": join,
        "max_ratio_pm3s": near[worst],
        "max_ratio_second": worst,
        "transient_seconds_pm3s": int(sum(r > RATIO_FLAG for r in near.values())),
        "cycles_within_half_fir": len(early),
        "cycles_within_half_fir_kept": int(early["cycle_id"].isin(kept["cycle_id"]).sum()),
        "bads": " ".join(bads),
        "bads_join_masked": " ".join(bads_masked),
        "bads_change": set(bads) != set(bads_masked),
        "join_epoch": epoch,
        "join_epoch_drop_log": drop,
    } | reference_contrasts(subject, dirs, anchors, phase_idx)


def multiverse_lines(table: pd.DataFrame, dirs: dict) -> list[str]:
    """Early-cycle removal per 0.5 Hz universe, cached subjects with the largest ratio."""
    cached = [s for s in table.sort_values("max_ratio_pm3s", ascending=False)["subject"]
              if any((config.DIR_MULTIVERSE / "branches" / s).glob("*"))]  # fmt: skip
    anchors = load_group_anchors(dirs["ersp"])
    lines = [
        "",
        f"Multiverse, {MULTIVERSE_HIGHPASS_HZ:g} Hz universes (float32 caches), Cz-ROI contrast "
        f"(dB) all cycles -> without cycles < {EARLY_S:g} s; subjects = the "
        f"{N_MULTIVERSE_SUBJECTS} cached subjects with the largest join ratio:",
    ]
    for s in cached[:N_MULTIVERSE_SUBJECTS]:
        cycles = pd.read_csv(dirs["gait_events"] / f"sub-{s}_cycles.tsv", sep="\t")
        late = cycles[cycles["rhs_start_s"] >= EARLY_S]
        for asr in config.MULTIVERSE_GRID["asr_mode"]:
            decisions = {"highpass_hz": MULTIVERSE_HIGHPASS_HZ, "asr_mode": asr,
                         "lowpass_hz": float(config.MULTIVERSE_LOWPASS_HZ)}  # fmt: skip
            for rule in config.MULTIVERSE_GRID["iclabel_rule"]:
                path = mp.branch_dir_for(s, decisions) / f"sub-{s}_desc-icaClean_{rule}_raw.fif"
                raw = mne.io.read_raw_fif(path, preload=True)
                w = linear_roi_weights(raw.info, center_ch=config.ROI_CENTER_CH)
                full, cut = (beta_phase_maps(raw, s, c, anchors) for c in (cycles, late))
                c_full = float(w @ (full[0] - full[1]).mean(1))
                c_cut = float(w @ (cut[0] - cut[1]).mean(1))
                lines.append(
                    f"  sub-{s} ASR {asr:<4s} {rule:<12s}: {c_full:+.3f} -> {c_cut:+.3f} "
                    f"(change {c_cut - c_full:+.3f}; cycles {full[2]} -> {cut[2]})"
                )
    return lines


def group_lines(table: pd.DataFrame) -> list[str]:
    """Group paired t-test (DS - swing against 0) stored vs without early cycles."""
    lines = ["", f"Group Cz-ROI beta double stance vs swing, n = {len(table)}:"]
    for col, label in (("stored_db", "stored"), ("without_early_db", f"without < {EARLY_S:g} s")):
        t = ttest_1samp(table[col], 0.0)
        lines.append(
            f"  {label:<16s}: t({len(table) - 1}) = {t.statistic:.3f}  p = {t.pvalue:.3g}  "
            f"mean = {table[col].mean():+.3f} dB"
        )
    change = table["without_early_db"] - table["stored_db"]
    i = change.abs().idxmax()
    lines.append(
        f"  largest per-subject change: sub-{table.subject[i]} {change[i]:+.3f} dB; "
        f"max |recomputed - stored| = {(table.recomputed_db - table.stored_db).abs().max():.1e} dB"
    )
    return lines


def main() -> None:
    """Measure every reference subject, then the multiverse subjects, and write both files."""
    dirs = get_dataset_dirs("stepup")
    anchors = load_reference_anchors(dirs["ersp"])
    phase_idx = phase_split_indices(anchors, n_points=config.TFR_N_POINTS)
    rows = []
    for s in reference_cohort("stepup", dirs):
        rows.append(subject_row(s, dirs, anchors, phase_idx))
        print(f"sub-{s} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(config.DIR_QC / "stepup_join_transient.tsv", sep="\t", index=False,
                 float_format="%.4g", lineterminator="\n")  # fmt: skip
    sf = config.TARGET_SFREQ
    lines = [
        "TREADMILL STAND/CS JOIN (concatenated before filtering)",
        "=" * 55,
        f"High-pass {config.REFERENCE_HIGHPASS_HZ:g} Hz FIR length {highpass_fir_s(sf):.2f} s "
        f"at {sf:g} Hz; ratio = per-second Cz-ROI variance / median of {REF_S[0]}-{REF_S[1]} s "
        f"either side; transient second = ratio > {RATIO_FLAG:g}",
        "",
        table.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    ]
    lines += group_lines(table)
    lines += [
        "",
        f"Bad-channel set changes with +/-{MASK_S:g} s masked: "
        f"{int(table.bads_change.sum())}/{len(table)} subjects; join epoch kept for the ICA fit: "
        f"{int((table.join_epoch_drop_log == '').sum())}/{len(table)}",
    ]
    lines += multiverse_lines(table, dirs)
    (config.DIR_QC / "stepup_join_transient.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
