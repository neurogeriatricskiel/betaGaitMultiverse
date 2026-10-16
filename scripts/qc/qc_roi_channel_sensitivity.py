"""
qc_roi_channel_sensitivity.py -- ROI weights, channel-level beta contrast, ROI sensitivity.

Purpose
    On the reference cohort (treadmill n = 17, overground primary n = 14):
    1. ROI weights of Fp1/Fpz/Fp2/AF3/AF4/AF7/AF8, the Fp*/AF* share of the
       total weight, and the five largest weights with their distance to Cz.
    2. Per-channel 13-30 Hz double stance minus swing (group mean, paired t),
       each channel's contribution w_i x c_i to the ROI contrast and the
       Fp*/AF* share of it.
    3. The ROI statistic without Fp*/AF*, also without F7/F8/FT*/T*/TP*, and
       for the five channels nearest Cz (equal weights); weights renormalised.
    5. Double stance, swing and their difference by frequency at Fpz and Cz.
    The stored ROI is reproduced first (sum w_i c_i = stored contrast).
Inputs
    datasets/<ds>/d05_ersp/sub-*_ersp_double_stance.npy, _swing.npy, _roi_weights.npy
    datasets/<ds>/d03_clean/sub-*_desc-icaClean_concat_raw.fif (channel names, positions)
Outputs
    results/pipeline/<ds>/qc/<ds>_roi_channel_sensitivity.txt
    results/pipeline/<ds>/qc/<ds>_channel_contrast.tsv, <ds>_freq_profile_fpz_cz.png
    results/supplementary/<ds>_roi_sensitivity.tsv, <ds>_contrast_topography.png
Manuscript
    Supplement (ROI sensitivity, contrast topography); Discussion (frontal channels).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/qc/qc_roi_channel_sensitivity.py  (< 1 min)
"""

import matplotlib
import numpy as np
from scipy.stats import ttest_1samp

from src import roi_diagnostics as rd
from src.config import DATASET, DIR_QC, DIR_SUPPLEMENTARY
from src.paths import get_dataset_dirs
from src.reference_stats import paired_test, phase_beta_values
from src.roi_figures import draw_contrast_topography, draw_freq_profile

FRONTAL_LISTED = ("Fp1", "Fpz", "Fp2", "AF3", "AF4", "AF7", "AF8")
ROI_LABELS = {
    "stored": "stored linear ROI (check)",
    "without_frontal": "without Fp*/AF*",
    "without_frontal_lateral": "without Fp*/AF*, F7/F8, FT*, T*, TP*",
    "narrow": "narrow ROI, five channels nearest Cz, equal weights",
}


def joined(items) -> str:
    """Comma-separated report items."""
    return ", ".join(items)


def weight_lines(maps: rd.PhaseMaps, fr: np.ndarray) -> list[str]:
    """Item 1: listed frontal weights, Fp*/AF* weight share, five largest weights."""
    ch, w = np.array(maps.ch_names), maps.weights
    lines = ["", "1. ROI weights", f"  Weights identical across subjects: {np.allclose(w, w[0])}"]
    for c in FRONTAL_LISTED:
        if c in maps.ch_names:
            lines.append(f"  {c:<4} w = {w[:, maps.ch_names.index(c)].mean():.4f}")
        else:
            lines.append(f"  {c:<4} not recorded")
    share = 100 * w[:, fr].sum(1).mean()
    lines.append(f"  Share of total ROI weight on Fp*/AF* ({joined(ch[fr])}): {share:.2f} %")
    d = rd.distances_to_cz(maps.infos[0])
    top = np.argsort(-w.mean(0))[:5]
    items = (f"{ch[i]} {w[:, i].mean():.4f} ({d[i]:.1f} mm from Cz)" for i in top)
    lines.append("  Five largest weights: " + joined(items))
    return lines


def contrast_lines(maps: rd.PhaseMaps, c: np.ndarray, fr: np.ndarray, lat: np.ndarray) -> list:
    """Item 2: largest channel contrasts and t, contributions and their Fp*/AF* share."""
    ch, m = np.array(maps.ch_names), c.mean(0)
    t = ttest_1samp(c, 0, axis=0)[0]
    contrib = (maps.weights * c).mean(0)
    total, share = contrib.sum(), 100 * maps.weights[:, fr].sum(1).mean()
    by_c = (f"{ch[i]} {m[i]:+.2f} dB (t={t[i]:.2f})" for i in np.argsort(-m)[:10])
    by_t = (f"{ch[i]} t={t[i]:.2f} ({m[i]:+.2f} dB)" for i in np.argsort(-t)[:10])
    by_w = (f"{ch[i]} {contrib[i]:+.3f}" for i in np.argsort(-np.abs(contrib))[:10])
    return [
        "",
        "2. Channel-level beta contrast (double stance - swing, 13-30 Hz), group mean and paired t",
        "  Ten largest contrast: " + joined(by_c),
        "  Ten largest t:        " + joined(by_t),
        f"  Contribution w_i x c_i (group mean), sums to {total:+.3f} dB = ROI mean contrast",
        "  Largest contributions: " + joined(by_w),
        f"  Fp*/AF* contribution: {contrib[fr].sum():+.3f} dB = "
        f"{100 * contrib[fr].sum() / total:.1f} % of the ROI contrast "
        f"(their weight share {share:.2f} %)",
        f"  F7/F8/FT*/T*/TP* contribution: {contrib[lat].sum():+.3f} dB = "
        f"{100 * contrib[lat].sum() / total:.1f} %",
    ]


def roi_lines(maps: rd.PhaseMaps, variants: dict, tests: dict) -> list[str]:
    """Item 3: one line per ROI variant (and the narrow ROI's channels)."""
    names = maps.ch_names
    removed = int((rd.channel_mask(names, rd.FRONTAL) | rd.channel_mask(names, rd.LATERAL)).sum())
    labels = dict(ROI_LABELS)
    labels["without_frontal_lateral"] += f" ({removed} ch)"
    narrow = [tuple(np.flatnonzero(row)) for row in variants["narrow"]]
    nearest = np.argsort(rd.distances_to_cz(maps.infos[0]))[: rd.N_NARROW]
    where = " (all subjects)" if len(set(narrow)) == 1 else " (subject 1)"
    lines = []
    for k, r in tests.items():
        if k == "narrow":
            lines.append(f"  Narrow ROI: {joined(np.array(maps.ch_names)[nearest])}{where}")
        lines.append(
            f"  {labels[k]:<55} t({r['df']}) = {r['t']:.3f}  p = {r['p']:.4g}  "
            f"mean = {r['mean']:+.3f} dB"
        )
    return lines


def write_tables(maps: rd.PhaseMaps, c: np.ndarray, tests: dict) -> None:
    """Per-channel table (QC) and the ROI sensitivity table (supplement)."""
    ch, d, m = maps.ch_names, rd.distances_to_cz(maps.infos[0]), c.mean(0)
    t, p = ttest_1samp(c, 0, axis=0)
    w, contrib = maps.weights.mean(0), (maps.weights * c).mean(0)
    rows = ["channel\tweight\tdist_cz_mm\tmean_contrast_db\tt\tp\tcontribution_db"]
    rows += [
        f"{ch[i]}\t{w[i]:.5f}\t{d[i]:.1f}\t{m[i]:.4f}\t{t[i]:.3f}\t{p[i]:.4g}\t{contrib[i]:.5f}"
        for i in np.argsort(-m)
    ]
    (DIR_QC / f"{DATASET}_channel_contrast.tsv").write_text("\n".join(rows) + "\n")
    rows = ["roi\tn_channels\tt\tdf\tp\tmean_db"]
    rows += [
        f"{k}\t{r['n_channels']}\t{r['t']:.4f}\t{r['df']}\t{r['p']:.4g}\t{r['mean']:.4f}"
        for k, r in tests.items()
    ]
    (DIR_SUPPLEMENTARY / f"{DATASET}_roi_sensitivity.tsv").write_text("\n".join(rows) + "\n")


def main() -> None:
    """Reproduce the stored ROI, then write every channel-level check of the active dataset."""
    matplotlib.use("Agg")
    dirs = get_dataset_dirs(DATASET)
    cohort = rd.reference_cohort(DATASET, dirs)
    maps = rd.load_phase_maps(cohort, dirs)
    c = rd.channel_contrast(maps)
    phases = [phase_beta_values(s, dirs["ersp"]) for s in cohort]
    ref = paired_test(*zip(*phases, strict=True))
    gap = np.abs((maps.weights * c).sum(1) - np.array([a - b for a, b in phases])).max()
    n, fr = len(cohort), rd.channel_mask(maps.ch_names, rd.FRONTAL)
    lines = [
        f"=== {DATASET}  n = {n}  ({joined(cohort)})",
        f"Stored reference: t({n - 1}) = {ref['t']:.3f}, p = {ref['p']:.4g}, "
        f"mean diff = {ref['mean_diff']:+.3f} dB",
        "Check: sum_i w_i * c_i reproduces each subject's stored ROI contrast, "
        f"max |diff| = {gap:.2e} dB",
    ]
    lines += weight_lines(maps, fr)
    lines += contrast_lines(maps, c, fr, rd.channel_mask(maps.ch_names, rd.LATERAL))
    variants = rd.roi_variants(maps)
    tests = {k: rd.roi_test(w, c) for k, w in variants.items()}
    lines += ["", "3. ROI sensitivity (weights renormalised to sum 1)"]
    lines += roi_lines(maps, variants, tests)
    lines += ["", "5. Frequency profile, group mean ERSP (dB) at Fpz and Cz"]
    lines += draw_freq_profile(maps, DATASET, DIR_QC / f"{DATASET}_freq_profile_fpz_cz.png")
    topo_path = DIR_SUPPLEMENTARY / f"{DATASET}_contrast_topography.png"
    draw_contrast_topography(c.mean(0), maps.infos[0], DATASET, n, topo_path)
    write_tables(maps, c, tests)
    (DIR_QC / f"{DATASET}_roi_channel_sensitivity.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
