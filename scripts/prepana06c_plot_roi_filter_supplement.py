"""
prepana06c_plot_roi_filter_supplement.py -- supplementary figure: ROI spatial-filter weights.

Purpose
    Topography of the linear Cz-centred ROI weights
    (src.spatial_filter.linear_roi_weights: w_i = 1 - d_i / d_max,
    normalised to sum 1, so weights fall linearly with distance from Cz to
    0 at the farthest channel). Methods figure, not a result. Weights are
    identical for every subject of a dataset (same positions) but differ
    between datasets: treadmill positions from the standard_1005 template,
    overground positions from each recording's BIDS electrodes.tsv. Also
    prints the lowest and highest weights with their distance to Cz.
Inputs
    datasets/<ds>/d03_clean/sub-*_desc-icaClean_concat_raw.fif (first subject found)
Outputs
    results/supplementary/<ds>_roi_filter_topo.png
Manuscript
    Methods 2.6 (ROI); supplementary figure.
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/prepana06c_plot_roi_filter_supplement.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

from src import plot_style as ps
from src.config import DATASET, DIR_SUPPLEMENTARY, ROI_CENTER_CH, SUBJECTS
from src.paths import get_dataset_dirs
from src.spatial_filter import TOPOMAP_SPHERE, linear_roi_weights


def first_clean_info(subjects: list[str], clean_dir: Path) -> tuple[str, mne.Info]:
    """Return the first subject with ICA-corrected data and its info (weights do not differ)."""
    for subject in subjects:
        p = clean_dir / f"sub-{subject}_desc-icaClean_concat_raw.fif"
        if p.exists():
            return subject, mne.io.read_raw_fif(p, preload=False, verbose=False).info
    raise SystemExit(f"No ICA-corrected data found for dataset={DATASET}")


def report_weights(info: mne.Info, weights: np.ndarray) -> None:
    """Print the lowest and highest weights with their distance to Cz."""
    ch_names = info["ch_names"]
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    dist_cm = np.linalg.norm(pos - pos[ch_names.index(ROI_CENTER_CH)], axis=1) * 100
    order = np.argsort(weights)
    print("Lowest 8 weights (channel, weight, distance to Cz in cm):")
    for i in order[:8]:
        print(f"  {ch_names[i]:6s}  {weights[i]:.5f}  {dist_cm[i]:.2f}")
    print("Highest 5 weights:")
    for i in order[::-1][:5]:
        print(f"  {ch_names[i]:6s}  {weights[i]:.5f}  {dist_cm[i]:.2f}")
    print(
        f"Zero weight: {[ch_names[i] for i in np.where(weights == 0)[0]]}  "
        f"(farthest channel: {ch_names[int(dist_cm.argmax())]})"
    )


def main() -> None:
    """Plot the ROI weight topography of the active dataset."""
    matplotlib.use("Agg")
    dirs = get_dataset_dirs(DATASET)
    used_subject, raw_info = first_clean_info(SUBJECTS, dirs["clean"])
    weights = linear_roi_weights(raw_info, center_ch=ROI_CENTER_CH)
    print(
        f"Dataset: {DATASET}, n_ch={len(raw_info['ch_names'])}, "
        f"representative subject sub-{used_subject}"
    )
    report_weights(raw_info, weights)

    ps.apply_rc()
    fig, ax = plt.subplots(figsize=ps.FIGSIZE["roi_topo"])
    fig.subplots_adjust(top=0.86)
    im, _ = mne.viz.plot_topomap(
        weights,
        raw_info,
        axes=ax,
        show=False,
        cmap=ps.WEIGHT_CMAP,
        vlim=(0, weights.max()),
        contours=4,
        sphere=TOPOMAP_SPHERE,
    )
    # n counts channels here: the weights are the same for every subject.
    ps.add_header(
        fig,
        f"{ps.DATASET_NAME[DATASET]} walking: ROI spatial-filter weights "
        f"(n = {len(raw_info['ch_names'])} channels)",
        f"Linear weights centred on {ROI_CENTER_CH}, identical for every subject",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ROI weight", fontsize=ps.FS_LABEL)
    cbar.ax.tick_params(labelsize=ps.FS_TICK)
    ps.save(fig, Path(DIR_SUPPLEMENTARY) / f"{DATASET}_roi_filter_topo.png")


if __name__ == "__main__":
    main()
