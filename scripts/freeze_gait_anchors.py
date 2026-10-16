"""
freeze_gait_anchors.py -- compute and freeze the group gait-event anchors (run by hand).

Purpose
    Group-median LTO/LHS/RTO positions (% of the RHS-RHS cycle), pooled
    over every kept cycle of every subject with a cycles_kept.tsv, written
    to the frozen calibration file read by src.ersp.load_group_anchors
    (treadmill reference warp and both multiverses). No pipeline stage
    recomputes or overwrites this file. Shows the change against the
    existing file and asks before overwriting.
    Frozen values: treadmill 10.0 / 50.0 / 59.8 %, overground
    16.0 / 50.3 / 66.0 % (overground reference uses fixed 18/50/68 %).
Inputs
    datasets/<ds>/d04_gaitepochs/sub-*_cycles_kept.tsv
Outputs
    datasets/<ds>/d05_ersp/group_gait_event_anchors_frozen.json
Manuscript
    Methods 2.5 (time warping).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/freeze_gait_anchors.py
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASET, SUBJECTS
from src.paths import get_dataset_dirs

FROZEN_NOTE = (
    "FROZEN calibration input. Computed once from the reference pipeline's kept gait "
    "cycles, pooled across every subject listed in subjects_pooled. No pipeline code "
    "path overwrites it (src.ersp.load_group_anchors raises if it is missing). To "
    "recompute, run scripts/freeze_gait_anchors.py by hand."
)


def pooled_event_fractions(subjects: list[str], gaitepoch_dir: Path):
    """Per-cycle LTO/LHS/RTO fractions of cycle duration, pooled over subjects."""
    f_lto, f_lhs, f_rto, pooled = [], [], [], []
    for subject in subjects:
        meta_path = gaitepoch_dir / f"sub-{subject}_cycles_kept.tsv"
        if not meta_path.exists():
            print(f"  sub-{subject}: no cycles_kept.tsv -- skipping")
            continue
        meta = pd.read_csv(meta_path, sep="\t")
        if len(meta) == 0:
            print(f"  sub-{subject}: cycles_kept.tsv is empty -- skipping")
            continue
        dur = meta["rhs_end_s"] - meta["rhs_start_s"]
        f_lto.append(((meta["lto_s"] - meta["rhs_start_s"]) / dur).values)
        f_lhs.append(((meta["lhs_s"] - meta["rhs_start_s"]) / dur).values)
        f_rto.append(((meta["rto_s"] - meta["rhs_start_s"]) / dur).values)
        pooled.append(subject)
    if not f_lto:
        raise RuntimeError("No sub-*_cycles_kept.tsv files found -- run prepana01-04 first.")
    return f_lto, f_lhs, f_rto, pooled


def group_anchors(subjects: list[str], gaitepoch_dir: Path) -> dict:
    """Group-median anchors (% of the cycle) and what they were pooled from."""
    f_lto, f_lhs, f_rto, pooled = pooled_event_fractions(subjects, gaitepoch_dir)
    a_lto, a_lhs, a_rto = (float(np.median(np.concatenate(f))) * 100 for f in (f_lto, f_lhs, f_rto))
    if not 0 < a_lto < a_lhs < a_rto < 100:
        raise RuntimeError(f"Group anchors are not monotonic: {a_lto}, {a_lhs}, {a_rto}")
    return {
        "A_lto_pct": a_lto,
        "A_lhs_pct": a_lhs,
        "A_rto_pct": a_rto,
        "n_subjects_pooled": len(pooled),
        "n_cycles_pooled": sum(len(a) for a in f_lto),
        "subjects_pooled": pooled,
    }


def confirm_overwrite(frozen_path: Path, new: dict) -> bool:
    """Show the existing frozen values and the change; ask before overwriting."""
    if not frozen_path.exists():
        print(f"\nNo existing frozen file at {frozen_path} -- creating it.")
        return True
    old = json.loads(frozen_path.read_text())
    print(f"\nExisting frozen file ({frozen_path.name}):")
    print(
        f"  A_lto={old['A_lto_pct']:.4f}%  A_lhs={old['A_lhs_pct']:.4f}%  "
        f"A_rto={old['A_rto_pct']:.4f}%  n={old['n_subjects_pooled']}  "
        f"frozen_date={old.get('frozen_date', '?')}"
    )
    print(
        "  Diff: "
        + "  ".join(
            f"d{k[:5]}={new[k] - old[k]:+.4f}" for k in ("A_lto_pct", "A_lhs_pct", "A_rto_pct")
        )
    )
    answer = input(f"\nOverwrite {frozen_path} with the new values above? [y/N] ")
    return answer.strip().lower() == "y"


def main() -> None:
    """Compute the anchors, show them and freeze them after confirmation."""
    dirs = get_dataset_dirs(DATASET)
    frozen_path = dirs["ersp"] / "group_gait_event_anchors_frozen.json"
    new = group_anchors(SUBJECTS, dirs["gaitepochs"])
    print(
        f"New anchors (pooled across {new['n_subjects_pooled']} subjects, "
        f"{new['n_cycles_pooled']} cycles):"
    )
    print(
        "  " + "  ".join(f"{k[:5]}={new[k]:.4f}%" for k in ("A_lto_pct", "A_lhs_pct", "A_rto_pct"))
    )
    print(f"  subjects: {new['subjects_pooled']}")
    if not confirm_overwrite(frozen_path, new):
        print("Not overwritten. Exiting.")
        return
    frozen = {**new, "frozen_date": date.today().isoformat(), "note": FROZEN_NOTE}
    frozen_path.write_text(json.dumps(frozen, indent=2))
    print(f"\nSaved -> {frozen_path}")


if __name__ == "__main__":
    main()
