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
        the linear Cz ROI, 1 - after / before; absolute power before and after
        (V^2/Hz) is kept in the TSV.
    Join-free rows (the STAND/CS join carries a filter transient,
    scripts/qc/qc_join_transient.py): standing without its last JOIN_TRIM_S
    and walking without its first JOIN_TRIM_S, each with Welch average
    "mean" and "median"; and the same from separate GEDAI fits on the
    standing and the walking recording alone (fit = "separate").
    Each row has a role:
      primary     -- the gating measure: walking without its first
                     JOIN_TRIM_S, Welch average "median", GEDAI fitted on the
                     whole recording, broadband multiplier 3;
      sensitivity -- every other join-free or separate-fit row (separate fit,
                     multiplier GEDAI_SENSITIVITY_BROADBAND_MULTIPLIER, mean
                     Welch, standing without its last JOIN_TRIM_S); not gating;
      superseded  -- the original whole / walking / standing rows (mean Welch,
                     whole-recording fit), kept for traceability; the
                     whole-recording figure is inflated by the join transient.
    Gate (src.config.GEDAI_GATE_THRESHOLD_PCT, our criterion): mean of the
    primary measure over GEDAI_GATE_SUBJECTS.
Inputs
    datasets/stepup/d00_raw/ (BrainVision STAND + CS)
Outputs
    results/pipeline/stepup/qc/stepup_gedai_gate_check.txt, stepup_gedai_gate.tsv
Manuscript
    Methods 2.7 (GEDAI exclusion).
Run
    python scripts/qc/qc_gedai_gate.py [all]   (needs the gedai extra; "all" = every
    reference subject instead of GEDAI_GATE_SUBJECTS; about 60 min per subject)
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
JOIN_FREE = ("standing_trim", "walking_trim")
ANNOTATION = {"walking": "CS", "standing": "STAND"}
JOIN_TRIM_S = 20.0  # s -- removed next to the join in the join-free rows
PRIMARY = {"segment": "walking_trim", "fit": "whole", "welch_average": "median"}
# (label, selection among sensitivity rows, compared with the limit); multiplier 3 unless set
SENSITIVITY = (
    ("separate fit", {"segment": "walking_trim", "fit": "separate", "welch_average": "median"},
     True),
    ("multiplier 6", {"segment": "walking_trim", "fit": "whole", "welch_average": "median",
                      "broadband_multiplier": config.GEDAI_SENSITIVITY_BROADBAND_MULTIPLIER},
     True),
    ("mean Welch", {"segment": "walking_trim", "fit": "whole", "welch_average": "mean"}, True),
    ("standing without last 20 s", {"segment": "standing_trim", "fit": "whole",
                                    "welch_average": "median"}, False),
)  # fmt: skip


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
    """Return the whole recording, a segment, or a segment without JOIN_TRIM_S at the join."""
    if name == "whole":
        return raw
    return trimmed(annotation_segment(raw, ANNOTATION[name.removesuffix("_trim")]), name)


def trimmed(raw, name: str):
    """Return a segment recording, without JOIN_TRIM_S at the join for the *_trim names."""
    if name == "standing_trim":
        return raw.copy().crop(tmax=raw.times[-1] - JOIN_TRIM_S)
    return raw.copy().crop(tmin=JOIN_TRIM_S) if name == "walking_trim" else raw


def removed(before, after, average: str = "mean") -> dict:
    """Return the removed fractions (%) and absolute ROI beta between two versions of a segment."""
    a, b = after.get_data(picks="eeg"), before.get_data(picks="eeg")
    weights = linear_roi_weights(before.info, center_ch=config.ROI_CENTER_CH)
    psd = [
        r.compute_psd(
            fmin=BETA_FMIN, fmax=BETA_FMAX, picks="eeg", average=average, verbose=False
        ).get_data()
        for r in (before, after)
    ]
    beta_b, beta_a = (float(apply_linear_roi(p, weights).mean()) for p in psd)
    return {
        "duration_s": float(before.times[-1]),
        "var_removed_pooled_pct": 100 * (1 - a.var() / b.var()),
        "var_removed_channel_mean_pct": 100 * float(np.mean(1 - a.var(1) / b.var(1))),
        "roi_beta_removed_pct": 100 * (1 - beta_a / beta_b),
        "roi_beta_before": beta_b,
        "roi_beta_after": beta_a,
    }


def row(subject: str, mult: float, name: str, fit: str, average: str) -> dict:
    """Return the identifying columns of one table row."""
    return {
        "subject": subject, "broadband_multiplier": mult, "segment": name, "fit": fit,
        "welch_average": average,
    }  # fmt: skip


def subject_rows(subject: str, raw_dir) -> list[dict]:
    """All segments x both broadband multipliers, then the join-free and separate-fit rows."""
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
            rows.append(row(subject, mult, name, "whole", "mean") | stats)
        for name in JOIN_FREE:
            for avg in ("mean", "median"):
                stats = removed(segment(before, name), segment(after, name), avg)
                rows.append(row(subject, mult, name, "whole", avg) | stats)
    del before, after
    mult = config.GEDAI_SETTINGS["broadband_noise_multiplier"]
    for seg_name, desc in ANNOTATION.items():
        seg = annotation_segment(raw, desc)
        seg.set_annotations(seg.annotations[[a["description"] == desc for a in seg.annotations]])
        before = preprocess_raw(seg.copy(), subject, use_gedai=False, **kw)
        after = preprocess_raw(seg.copy(), subject, use_gedai=True, **kw)
        for name in (seg_name, f"{seg_name}_trim"):
            for avg in ("mean", "median"):
                stats = removed(trimmed(before, name), trimmed(after, name), avg)
                rows.append(row(subject, mult, name, "separate", avg) | stats)
    return rows


def assign_role(table: pd.DataFrame) -> pd.Series:
    """Return primary / sensitivity / superseded for every row (see the module docstring)."""
    base = config.GEDAI_SETTINGS["broadband_noise_multiplier"]
    primary = (
        (table.segment == PRIMARY["segment"]) & (table.fit == PRIMARY["fit"])
        & (table.welch_average == PRIMARY["welch_average"])
        & (table.broadband_multiplier == base)
    )  # fmt: skip
    superseded = (
        table.segment.isin(SEGMENTS) & (table.fit == "whole") & (table.welch_average == "mean")
    )
    return pd.Series(
        np.select([primary, superseded], ["primary", "superseded"], "sensitivity"),
        index=table.index,
    )


def gate_value(table: pd.DataFrame, gate_subjects: list[str]) -> float:
    """Return the gate value: mean primary ROI beta removed (%) over the gate subjects."""
    rows = table[(table.role == "primary") & table.subject.isin(gate_subjects)]
    if rows.subject.nunique() != len(gate_subjects) or len(rows) != len(gate_subjects):
        raise ValueError("gate needs exactly one primary row per gate subject")
    return float(rows.roi_beta_removed_pct.mean())


def measure_line(label: str, g: pd.DataFrame) -> str:
    """Per-subject ROI beta removed, mean and range of one measure."""
    beta = g.roi_beta_removed_pct
    per = ", ".join(f"{s} {v:.2f}" for s, v in zip(g.subject, beta, strict=True))
    return f"  {label}: {per}; mean {beta.mean():.2f} % (range {beta.min():.2f}-{beta.max():.2f})"


def report(table: pd.DataFrame, gate_subjects: list[str], minutes: float) -> list[str]:
    """Return primary and sensitivity results, then provenance, the full table, superseded rows."""
    node = hashlib.sha256(open(gedai_node.__file__, "rb").read()).hexdigest()
    base = config.GEDAI_SETTINGS["broadband_noise_multiplier"]
    limit = config.GEDAI_GATE_THRESHOLD_PCT
    gate = gate_value(table, gate_subjects)
    ok = gate <= limit
    on = table[table.subject.isin(gate_subjects)]
    show = table.drop(columns=["roi_beta_before", "roi_beta_after", "duration_s"])
    lines = [
        "GEDAI GATE CHECK (treadmill)",
        "=" * 55,
        f"Primary (gating): % Cz-ROI 13-30 Hz power removed, walking without its first "
        f"{JOIN_TRIM_S:g} s, Welch average median, GEDAI fitted on the whole recording, "
        f"broadband multiplier {base:g}; mean over {', '.join(gate_subjects)}; limit {limit:g} % "
        "(our criterion, not from the literature)",
        measure_line("primary", on[on.role == "primary"]),
        f"Gate value {gate:.2f} % -> {'PASS' if ok else 'FAIL'}: GEDAI "
        f"{'viable' if ok else 'NOT viable'} as a multiverse node",
        "",
        "Sensitivity (not gating):",
    ]
    differ = []
    for label, sel, decides in SENSITIVITY:
        sel = sel | {"broadband_multiplier": sel.get("broadband_multiplier", base)}
        g = on[(on.role == "sensitivity") & np.logical_and.reduce(
            [on[k] == v for k, v in sel.items()]
        )]  # fmt: skip
        if decides and (g.roi_beta_removed_pct.mean() <= limit) != ok:
            differ.append(label)
        lines.append(measure_line(label, g))
    lines += [
        "",
        "The primary measure was defined after a join transient inflated the original "
        "whole-recording figure (see stepup_join_transient.txt); "
        + (
            "the sensitivity analysis gives the same decision."
            if not differ
            else "the sensitivity analysis does NOT give the same decision for: "
            + ", ".join(differ)
            + "."
        ),
        "(Decision compared for the walking sensitivity measures; standing without its last "
        f"{JOIN_TRIM_S:g} s shows removal at rest and is not compared with the limit.)",
        "",
    ]
    lines += [
        f"gedai package {importlib.metadata.version('gedai')}; "
        f"src/nodes/gedai_node.py sha256 {node}",
        f"Settings: {config.GEDAI_SETTINGS}; sensitivity broadband multiplier "
        f"{config.GEDAI_SENSITIVITY_BROADBAND_MULTIPLIER:g}",
        f"Filters (treadmill reference): high-pass {config.REFERENCE_HIGHPASS_HZ:g} Hz, low-pass "
        f"{config.REFERENCE_LOWPASS_HZ:g} Hz, ASR {'on' if config.REFERENCE_USE_ASR else 'off'}; "
        "GEDAI before the average reference",
        "Metrics: % removed = 100 * (1 - after / before); variance pooled over channels and "
        "samples, and mean of per-channel fractions; ROI beta = linear Cz ROI of the Welch PSD, "
        "13-30 Hz; fit = GEDAI on the whole recording or on each segment alone; *_trim = "
        f"without {JOIN_TRIM_S:g} s at the join; absolute ROI beta before / after (V^2/Hz) "
        "in stepup_gedai_gate.tsv",
        "Superseded rows (role = superseded): the original whole / walking / standing rows, "
        "mean Welch; the whole-recording figure is inflated by the join transient and is not "
        "a result",
        f"Subjects: {', '.join(table.subject.unique())}  ({minutes:.0f} min)",
        "",
        show.to_string(index=False, float_format=lambda v: f"{v:.2f}"),
        "",
    ]
    keys = ["role", "segment", "broadband_multiplier", "fit", "welch_average"]
    for (role, seg, mult, fit, avg), g in table.groupby(keys, sort=False):
        beta = g.roi_beta_removed_pct
        lines.append(
            f"Mean [{role}] {seg:<13} multiplier {mult:g} fit {fit:<8} {avg:<6}: variance "
            f"pooled {g.var_removed_pooled_pct.mean():.2f} %, "
            f"per channel {g.var_removed_channel_mean_pct.mean():.2f} %, ROI beta "
            f"{beta.mean():.2f} % (range {beta.min():.2f}-{beta.max():.2f}, n = {len(g)})"
        )
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
    table.insert(1, "role", assign_role(table))
    table.to_csv(config.DIR_QC / "stepup_gedai_gate.tsv", sep="\t", index=False,
                 float_format="%.10g", lineterminator="\n")  # fmt: skip
    lines = report(table, gate_subjects, (time.time() - t0) / 60)
    (config.DIR_QC / "stepup_gedai_gate_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
