"""
qc_baseline_window.py -- per-subject check of the standing-baseline window.

Purpose
    Treadmill: duration of the STAND and CS recordings (BrainVision
    headers); baseline = STAND minus its last BASELINE_EDGE_TRIM_S.
    Overground: BASELINE_START_VALUE / BASELINE_END_VALUE (start_standing /
    end_standing) each present exactly once in events.tsv, window length,
    window ends before the walking segment (SEGMENT_START_VALUE); baseline
    = window minus BASELINE_EDGE_TRIM_S. The same window calibrates ASR.
    Header-only reads; writes nothing besides the table below.
Inputs
    datasets/<ds>/d00_raw/ (BrainVision headers or BIDS events.tsv)
Outputs
    results/pipeline/<ds>/qc/<ds>_baseline_window.tsv
Manuscript
    Methods 2.5 (standing baseline).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/qc/qc_baseline_window.py   (< 1 min)
"""

from pathlib import Path

import mne
import pandas as pd

from src import config
from src.config import BASELINE_EDGE_TRIM_S, DATASET, DIR_QC, SUBJECTS
from src.paths import get_dataset_dirs


def treadmill_rows(raw_dir: Path) -> list[dict]:
    """Standing and walking recording lengths and the baseline used, per treadmill subject."""
    rows = []
    for s in SUBJECTS:
        dur = {}
        for task in (config.TASK_STAND, config.TASK_WALK):
            raw = mne.io.read_raw_brainvision(
                raw_dir / f"sub-{s}" / "eeg" / f"sub-{s}_task-{task}.vhdr",
                preload=False,
                verbose=False,
            )
            dur[task] = raw.n_times / raw.info["sfreq"]
        rows.append(
            {
                "subject": s,
                "stand_s": round(dur[config.TASK_STAND], 1),
                "walk_s": round(dur[config.TASK_WALK], 1),
                "baseline_used_s": round(dur[config.TASK_STAND] - BASELINE_EDGE_TRIM_S, 1),
                "ok": dur[config.TASK_STAND] > BASELINE_EDGE_TRIM_S,
            }
        )
    return rows


def event_onsets(events: pd.DataFrame, value: str) -> list[float]:
    """Onsets (s) of every event with this value."""
    return events.loc[events["value"] == value, "onset"].astype(float).tolist()


def overground_rows(raw_dir: Path) -> list[dict]:
    """Standing window from the BIDS events and the baseline used, per overground subject."""
    rows = []
    for s in SUBJECTS:
        ev = pd.read_csv(
            raw_dir / f"sub-{s}" / "eeg" / f"sub-{s}_task-{config.TASK_NAME}_events.tsv", sep="\t"
        )
        start = event_onsets(ev, config.BASELINE_START_VALUE)
        end = event_onsets(ev, config.BASELINE_END_VALUE)
        walk = event_onsets(ev, config.SEGMENT_START_VALUE)
        once = len(start) == 1 and len(end) == 1 and len(walk) == 1
        window = end[0] - start[0] if once else None
        rows.append(
            {
                "subject": s,
                "n_start_standing": len(start),
                "n_end_standing": len(end),
                "baseline_window_s": round(window, 1) if once else None,
                "baseline_used_s": round(window - BASELINE_EDGE_TRIM_S, 1) if once else None,
                "ends_before_walking": (end[0] <= walk[0]) if once else None,
                "gap_to_walking_s": round(walk[0] - end[0], 1) if once else None,
                "ok": bool(once and window > BASELINE_EDGE_TRIM_S and end[0] <= walk[0]),
            }
        )
    return rows


def main() -> None:
    """Write the baseline-window table of the active dataset."""
    raw_dir = get_dataset_dirs(DATASET)["raw"]
    rows = treadmill_rows(raw_dir) if DATASET == "stepup" else overground_rows(raw_dir)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    used = df["baseline_used_s"].dropna()
    print(
        f"\nBaseline used: {used.min():.1f}-{used.max():.1f} s (median {used.median():.1f}); "
        f"all ok: {bool(df['ok'].all())}"
    )
    out = DIR_QC / f"{DATASET}_baseline_window.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
