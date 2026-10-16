"""
qc_heelstrike_locked.py -- heel-strike-locked average at Fpz and Cz (movement-artefact check).

Purpose
    Group mean of the 1-40 Hz walking signal locked to every right and left
    heel strike of the kept cycles (-200 to +400 ms, baseline -200 to 0 ms),
    at Fpz and Cz, before ICA (stage-2 data) and after ICA (the data the
    ERSP uses). Reports the peak-to-peak deflection and the largest absolute
    value after heel strike, per channel and input.
Inputs
    datasets/<ds>/d02_prep/sub-*_concat_raw.fif, d03_clean/sub-*_desc-icaClean_concat_raw.fif
    datasets/<ds>/d04_gaitepochs/sub-*_cycles_kept.tsv (heel strikes, s from walk onset)
Outputs
    results/pipeline/<ds>/qc/<ds>_heelstrike_locked.txt, .png
Manuscript
    Discussion (frontal channels, movement artefact).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/qc/qc_heelstrike_locked.py   (~2 min)
"""

import matplotlib
import mne
import numpy as np
import pandas as pd

from src import roi_diagnostics as rd
from src.config import DATASET, DIR_QC
from src.paths import get_dataset_dirs
from src.roi_figures import draw_heelstrike

INPUTS = {
    "pre-ICA": ("prep", "_concat_raw.fif"),
    "ICA-corrected": ("clean", "_desc-icaClean_concat_raw.fif"),
}
CHANNELS = ["Fpz", "Cz"]


def summary_lines(averages: dict, times_ms: np.ndarray) -> list[str]:
    """Peak-to-peak and largest absolute value after heel strike, per input and channel."""
    lines = []
    after = times_ms >= 0
    for key, group in averages.items():
        for row, channel in enumerate(CHANNELS):
            post = group[row][after]
            peak = times_ms[after][np.argmax(np.abs(post))]
            p2p = post.max() - post.min()
            lines.append(
                f"  {key:<14} {channel:<4} peak-to-peak 0..400 ms {p2p:6.2f} uV  "
                f"max |x| {np.abs(post).max():6.2f} uV at {peak:.0f} ms"
            )
    return lines


def main() -> None:
    """Average every subject's heel-strike-locked signal and write the report and plot."""
    matplotlib.use("Agg")
    dirs = get_dataset_dirs(DATASET)
    cohort = rd.reference_cohort(DATASET, dirs)
    per_subject = {key: [] for key in INPUTS}
    for s in cohort:
        cycles = pd.read_csv(dirs["gaitepochs"] / f"sub-{s}_cycles_kept.tsv", sep="\t")
        for key, (folder, suffix) in INPUTS.items():
            raw = mne.io.read_raw_fif(
                dirs[folder] / f"sub-{s}{suffix}", preload=True, verbose=False
            )
            times_ms, avg = rd.heelstrike_average(raw, rd.heel_strike_times(cycles), CHANNELS)
            per_subject[key].append(avg)
        print(f"sub-{s} done", flush=True)
    averages = {key: np.mean(arrs, axis=0) for key, arrs in per_subject.items()}
    lines = [
        f"=== {DATASET} (n = {len(cohort)}): heel-strike-locked group mean, 1-40 Hz, "
        "baseline -200..0 ms"
    ]
    lines += summary_lines(averages, times_ms)
    draw_heelstrike(
        averages, times_ms, DATASET, len(cohort), DIR_QC / f"{DATASET}_heelstrike_locked.png"
    )
    (DIR_QC / f"{DATASET}_heelstrike_locked.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
