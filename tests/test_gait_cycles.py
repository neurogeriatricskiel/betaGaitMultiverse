"""Gait events from a synthetic BIDS events table, including the IC_/TO_ synonym labels."""

import numpy as np
import pandas as pd

from src.gait_cycles import extract_valid_gait_cycles, load_gait_events_from_tsv

FS = 500.0
SYNONYMS = {
    "LHS": ["LeftHS", "IC_Left"],
    "RHS": ["RightHS", "IC_Right"],
    "LTO": ["LeftTO", "TO_Left"],
    "RTO": ["RightTO", "TO_Right"],
}


def write_events(path, labels, n_cycles=6, cycle_s=1.0, seg_start_s=10.0):
    """Regular gait: per cycle RHS 0 %, LTO 16 %, LHS 50 %, RTO 66 %."""
    rows = [("start_easy", seg_start_s), ("start_standing", 1.0), ("end_standing", 5.0)]
    for k in range(n_cycles + 1):
        t0 = seg_start_s + 0.5 + k * cycle_s
        rows.append((labels["RHS"], t0))
        if k < n_cycles:
            rows += [
                (labels["LTO"], t0 + 0.16),
                (labels["LHS"], t0 + 0.50),
                (labels["RTO"], t0 + 0.66),
            ]
    rows.append(("end_easy", seg_start_s + 0.5 + (n_cycles + 1) * cycle_s))
    df = pd.DataFrame(rows, columns=["value", "onset"])
    df["sample"] = (df["onset"] * FS).round().astype(int)
    df.sort_values("onset").to_csv(path, sep="\t", index=False)


def test_events_and_cycles(tmp_path):
    path = tmp_path / "events.tsv"
    write_events(path, {"RHS": "RightHS", "LTO": "LeftTO", "LHS": "LeftHS", "RTO": "RightTO"})
    lhs, lto, rhs, rto, t0, t1 = load_gait_events_from_tsv(path, "start_easy", "end_easy", SYNONYMS)
    assert (len(rhs), len(lto), len(lhs), len(rto)) == (7, 6, 6, 6)
    assert t0 == 10.0
    assert rhs[0] == round(0.5 * FS)  # samples relative to the segment start

    cycles = extract_valid_gait_cycles(lhs, lto, rhs, rto, FS)
    assert len(cycles) == 6
    assert np.allclose(cycles["duration_s"], 1.0)
    assert np.allclose(cycles["lto_frac"], 0.16)


def test_synonym_labels_give_same_events(tmp_path):
    std, syn = tmp_path / "std.tsv", tmp_path / "syn.tsv"
    write_events(std, {"RHS": "RightHS", "LTO": "LeftTO", "LHS": "LeftHS", "RTO": "RightTO"})
    write_events(syn, {"RHS": "IC_Right", "LTO": "TO_Left", "LHS": "IC_Left", "RTO": "TO_Right"})
    a = load_gait_events_from_tsv(std, "start_easy", "end_easy", SYNONYMS)
    b = load_gait_events_from_tsv(syn, "start_easy", "end_easy", SYNONYMS)
    for x, y in zip(a[:4], b[:4], strict=True):
        assert np.array_equal(x, y)


def test_late_toe_off_rejected(tmp_path):
    path = tmp_path / "events.tsv"
    write_events(path, {"RHS": "RightHS", "LTO": "LeftTO", "LHS": "LeftHS", "RTO": "RightTO"})
    lhs, lto, rhs, rto, *_ = load_gait_events_from_tsv(path, "start_easy", "end_easy", SYNONYMS)
    lto_late = rhs[:-1] + int(0.40 * FS)  # LTO at 40 % > CYCLE_LTO_MAX_FRAC (30 %)
    assert len(extract_valid_gait_cycles(lhs, lto_late, rhs, rto, FS)) == 0
