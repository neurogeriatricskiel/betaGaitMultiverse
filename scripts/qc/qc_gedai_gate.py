"""
qc_gedai_gate.py -- GEDAI gate: how much variance and Cz-ROI beta power GEDAI removes (treadmill).

Purpose
    GEDAI (src/nodes/gedai_node.py) was evaluated as a candidate artefact-
    removal node and excluded. Per subject, the raw STAND + CS recording is
    preprocessed twice with the treadmill reference filters
    (src.pipeline_steps.preprocess_raw: REFERENCE_HIGHPASS_HZ,
    REFERENCE_LOWPASS_HZ, ASR as REFERENCE_USE_ASR), once without and once
    with GEDAI, so the only difference is the GEDAI step. Removed fraction,
    for the whole recording, the walking (CS) and the standing (STAND)
    segment:
      - total variance: pooled over channels and samples, and the mean of
        the per-channel fractions;
      - Cz-ROI beta power: Welch PSD (MNE defaults), 13-30 Hz, reduced with
        the linear Cz ROI, 1 - after / before.
    Gate (src.config.GEDAI_GATE_THRESHOLD_PCT, our criterion): mean ROI beta
    removed over GEDAI_GATE_SUBJECTS, whole recording, configured settings.
    Sensitivity: the node's own broadband multiplier
    (GEDAI_SENSITIVITY_BROADBAND_MULTIPLIER).
Inputs
    datasets/stepup/d00_raw/ (BrainVision STAND + CS)
Outputs
    results/pipeline/stepup/qc/stepup_gedai_gate_check.txt, stepup_gedai_gate.tsv
Manuscript
    Methods 2.7 (GEDAI exclusion).
Run
    python scripts/qc/qc_gedai_gate.py [all]   (needs the gedai extra; "all" = every
    reference subject instead of GEDAI_GATE_SUBJECTS; about 40 min per subject)
"""

import hashlib
import importlib.metadata
import inspect
import os
import sys
import time

os.environ["BETAGAIT_DATASET"] = "stepup"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.cycle_tfr import annotation_segment  # noqa: E402
from src.ersp import BETA_FMAX, BETA_FMIN  # noqa: E402
from src.nodes import gedai_node  # noqa: E402
from src.paths import get_dataset_dirs  # noqa: E402
from src.pipeline_steps import load_and_concatenate, preprocess_raw  # noqa: E402
from src.roi_diagnostics import reference_cohort  # noqa: E402
from src.spatial_filter import apply_linear_roi, linear_roi_weights  # noqa: E402

SEGMENTS = ("whole", "walking", "standing")
ANNOTATION = {"walking": "CS", "standing": "STAND"}


def check_settings() -> None:
    """Stop unless every configured GEDAI setting is what preprocess_raw actually applies."""
    defaults = {
        k: p.default for k, p in inspect.signature(gedai_node.apply_gedai_node).parameters.items()
    }
    for key, value in config.GEDAI_SETTINGS.items():
        if key != "broadband_noise_multiplier" and defaults[key] != value:
            raise SystemExit(f"GEDAI_SETTINGS[{key!r}] = {value!r}, node default {defaults[key]!r}")
    passed = inspect.signature(preprocess_raw).parameters["gedai_noise_multiplier"].default
    if passed != config.GEDAI_SETTINGS["broadband_noise_multiplier"]:
        raise SystemExit("broadband multiplier differs from preprocess_raw's default")


def segment(raw, name: str):
    """Return the whole recording or its walking / standing segment."""
    return raw if name == "whole" else annotation_segment(raw, ANNOTATION[name])


def removed(before, after) -> dict:
    """Return the removed fractions (%) between two versions of the same segment."""
    a, b = after.get_data(picks="eeg"), before.get_data(picks="eeg")
    weights = linear_roi_weights(before.info, center_ch=config.ROI_CENTER_CH)
    psd = [
        r.compute_psd(fmin=BETA_FMIN, fmax=BETA_FMAX, picks="eeg", verbose=False).get_data()
        for r in (before, after)
    ]
    beta_b, beta_a = (float(apply_linear_roi(p, weights).mean()) for p in psd)
    return {
        "var_removed_pooled_pct": 100 * (1 - a.var() / b.var()),
        "var_removed_channel_mean_pct": 100 * float(np.mean(1 - a.var(1) / b.var(1))),
        "roi_beta_removed_pct": 100 * (1 - beta_a / beta_b),
    }


def subject_rows(subject: str, raw_dir) -> list[dict]:
    """All segments x both broadband multipliers for one subject."""
    raw = load_and_concatenate(subject, raw_dir)
    kw = {
        "highpass_hz": config.REFERENCE_HIGHPASS_HZ, "lowpass_hz": config.REFERENCE_LOWPASS_HZ,
        "use_asr": config.REFERENCE_USE_ASR, "asr_cutoff": config.ASR_CUTOFF,
    }  # fmt: skip
    before = preprocess_raw(raw.copy(), subject, use_gedai=False, **kw)
    rows = []
    for mult in (config.GEDAI_SETTINGS["broadband_noise_multiplier"],
                 config.GEDAI_SENSITIVITY_BROADBAND_MULTIPLIER):  # fmt: skip
        after = preprocess_raw(
            raw.copy(), subject, use_gedai=True, gedai_noise_multiplier=mult, **kw
        )
        for name in SEGMENTS:
            stats = removed(segment(before, name), segment(after, name))
            rows.append({"subject": subject, "broadband_multiplier": mult, "segment": name} | stats)
    return rows


def report(table: pd.DataFrame, gate_subjects: list[str], minutes: float) -> list[str]:
    """Header with provenance and settings, the per-subject table and the gate result."""
    node = hashlib.sha256(open(gedai_node.__file__, "rb").read()).hexdigest()
    base = config.GEDAI_SETTINGS["broadband_noise_multiplier"]
    gate = table[(table.segment == "whole") & (table.broadband_multiplier == base)
                 & table.subject.isin(gate_subjects)]["roi_beta_removed_pct"].mean()  # fmt: skip
    ok = gate <= config.GEDAI_GATE_THRESHOLD_PCT
    lines = [
        "GEDAI GATE CHECK (treadmill)",
        "=" * 55,
        f"gedai package {importlib.metadata.version('gedai')}; "
        f"src/nodes/gedai_node.py sha256 {node}",
        f"Settings: {config.GEDAI_SETTINGS}; sensitivity broadband multiplier "
        f"{config.GEDAI_SENSITIVITY_BROADBAND_MULTIPLIER:g}",
        f"Filters (treadmill reference): high-pass {config.REFERENCE_HIGHPASS_HZ:g} Hz, low-pass "
        f"{config.REFERENCE_LOWPASS_HZ:g} Hz, ASR {'on' if config.REFERENCE_USE_ASR else 'off'}; "
        "GEDAI before the average reference",
        "Metrics: % removed = 100 * (1 - after / before); variance pooled over channels and "
        "samples, and mean of per-channel fractions; ROI beta = linear Cz ROI of the Welch PSD, "
        "13-30 Hz",
        f"Subjects: {', '.join(table.subject.unique())}  ({minutes:.0f} min)",
        "",
        table.to_string(index=False, float_format=lambda v: f"{v:.2f}"),
        "",
    ]
    for (seg, mult), g in table.groupby(["segment", "broadband_multiplier"], sort=False):
        lines.append(
            f"Mean {seg:<8} multiplier {mult:g}: variance pooled "
            f"{g.var_removed_pooled_pct.mean():.2f} %, "
            f"per channel {g.var_removed_channel_mean_pct.mean():.2f} %, ROI beta "
            f"{g.roi_beta_removed_pct.mean():.2f} %  (n = {len(g)})"
        )
    lines += [
        "",
        f"Gate (ours, not from the literature): mean ROI beta removed, whole recording, multiplier "
        f"{base:g}, {', '.join(gate_subjects)} <= {config.GEDAI_GATE_THRESHOLD_PCT:g} %",
        f"Gate value {gate:.2f} % -> {'PASS' if ok else 'FAIL'}: GEDAI "
        f"{'viable' if ok else 'NOT viable'} as a multiverse node",
    ]
    return lines


def main() -> None:
    """Run the check on the gate subjects (or every reference subject) and write both files."""
    check_settings()
    dirs = get_dataset_dirs("stepup")
    gate_subjects = list(config.GEDAI_GATE_SUBJECTS)
    subjects = reference_cohort("stepup", dirs) if sys.argv[1:] == ["all"] else gate_subjects
    t0, rows = time.time(), []
    for s in subjects:
        rows += subject_rows(s, dirs["raw"])
        print(f"sub-{s} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(config.DIR_QC / "stepup_gedai_gate.tsv", sep="\t", index=False,
                 float_format="%.4f", lineterminator="\n")  # fmt: skip
    lines = report(table, gate_subjects, (time.time() - t0) / 60)
    (config.DIR_QC / "stepup_gedai_gate_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
