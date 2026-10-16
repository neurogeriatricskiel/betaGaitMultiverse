"""Reviewer-facing QC tables of one dataset (scripts/qc/qc_build_tables.py). Read-only."""

import json
import types
from pathlib import Path

import mne
import pandas as pd

from src import config
from src.config import DATASET, DATASET_LABELS, DIR_QC, EPOCH_DUR, MULTIVERSE_SUBJECTS, SUBJECTS
from src.paths import get_dataset_dirs
from src.qc import DUPLICATE_REPORT, STAGE_ORDER, load_cohort_variants

FLAG_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def read_json(path: Path) -> dict:
    """JSON file contents, or {} if the file does not exist."""
    return json.loads(path.read_text()) if path.exists() else {}


def saved_file_counts(subject: str, dirs: dict) -> dict:
    """Channels analysed, applied high-pass and AutoReject epoch counts from saved files."""
    prep_path = dirs["prep"] / f"sub-{subject}_concat_raw.fif"
    epo_path = dirs["prep"] / f"sub-{subject}_preica_clean_epo.fif"
    clean_path = dirs["clean"] / f"sub-{subject}_desc-icaClean_concat_raw.fif"
    out = {"n_ch": None, "hp": None, "n_clean": None, "n_total": None}
    if clean_path.exists():
        out["n_ch"] = len(mne.io.read_raw_fif(clean_path, preload=False, verbose=False).ch_names)
    if prep_path.exists():
        raw = mne.io.read_raw_fif(prep_path, preload=False, verbose=False)
        out["hp"] = raw.info["highpass"]
        # Fixed-length epochs AutoReject saw (same call as fit_ica).
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DUR, preload=False, verbose=False)
        out["n_total"] = len(epochs.drop_bad(verbose=False))
    if epo_path.exists():
        out["n_clean"] = len(mne.read_epochs(epo_path, preload=False, verbose=False))
    return out


def subject_row(subject: str, dirs: dict) -> dict:
    """One row of <dataset>_subject_qc.tsv."""
    q = {st: read_json(dirs["qc"] / f"sub-{subject}_{st}_qc.json") for st in STAGE_ORDER}
    counts = saved_file_counts(subject, dirs)
    prep = q["preprocessing"]
    stale = prep and bool(prep.get("use_asr")) != config.REFERENCE_USE_ASR
    flags = [v.get("flag") for v in q.values() if v.get("flag")]
    worst = max(flags, key=lambda f: FLAG_ORDER.get(f, 0)) if flags else "missing"
    note = (
        "QC jsons are from an earlier run with a different ASR setting; the "
        "current reference run failed (see exclusions)"
    )
    return {
        "subject": subject,
        "n_channels_analysed": counts["n_ch"],
        "bad_channels_interpolated": "not recorded",
        "highpass_applied_hz": counts["hp"],
        "asr": f"SD {prep.get('asr_cutoff'):g}"
        if prep.get("use_asr")
        else ("off" if prep else None),
        "ica_components": q["ica"].get("n_total_comps"),
        "brain_ics_kept": q["ica"].get("n_brain_comps"),
        "autoreject_epochs_clean": counts["n_clean"],
        "autoreject_epochs_total": counts["n_total"],
        "cycles_detected": q["gait_events"].get("n_cycles"),
        "cycles_extracted": q["gait_epochs"].get("n_extracted"),
        "cycles_used_ersp": q["ersp"].get("n_cycles_used"),
        "in_reference": (dirs["ersp"] / f"sub-{subject}_ersp_beta.npy").exists(),
        "in_multiverse": subject in MULTIVERSE_SUBJECTS,
        "flag_worst_stage": worst,
        "flags": ";".join(f"{st}={v.get('flag', '-')}" for st, v in q.items()),
        "note": note if stale else "",
    }


def subject_table(dirs: dict) -> pd.DataFrame:
    """<dataset>_subject_qc.tsv: one row per configured subject."""
    return pd.DataFrame([subject_row(s, dirs) for s in SUBJECTS])


def treadmill_exclusions(dirs: dict) -> list[dict]:
    """S21 (reference and multiverse), S10 and S2 (multiverse only), with their numbers."""
    s21 = read_json(dirs["qc"] / "sub-S21_autoreject_recheck_qc.json")
    s10 = read_json(dirs["qc"] / "sub-S10_ica_qc.json")
    s2 = read_json(dirs["qc"] / "sub-S2_multiverse_branch_autoreject_check_qc.json")
    fails = [b for b in s2.get("branches", []) if b["outcome"] == "fail"]
    s2_numbers = "; ".join(
        f"HP {b['highpass_hz']} ASR {b['asr_mode']}: "
        f"{b['n_epochs_clean']}/{b['n_epochs_total']} clean"
        for b in fails
    )
    return [
        {
            "subject": "S21",
            "stage": "reference + multiverse",
            "reason": "AutoReject: fewer clean epochs than the 20 the ICA fit requires",
            "numbers": f"{s21.get('n_epochs_clean')}/{s21.get('n_epochs_total')} clean "
            f"(reject_frac {s21.get('reject_frac')}), ASR-off reference data",
            "source": "datasets/stepup/d00_qc/sub-S21_autoreject_recheck_qc.json "
            "(recheck script not preserved)",
        },
        {
            "subject": "S10",
            "stage": "multiverse only (kept in reference, n = 17)",
            "reason": "ICA: almost no brain components",
            "numbers": f"{s10.get('n_brain_comps')}/{s10.get('n_total_comps')} brain ICs "
            f"(brain_frac {s10.get('brain_frac')}) in the reference ICA",
            "source": "datasets/stepup/d00_qc/sub-S10_ica_qc.json",
        },
        {
            "subject": "S2",
            "stage": "multiverse only (kept in reference, n = 17)",
            "reason": "AutoReject fails in some multiverse branches; excluded from all universes "
            "as one subject-level decision",
            "numbers": s2_numbers
            + f" ({len(fails)} of {len(s2.get('branches', []))} branches fail)",
            "source": "datasets/stepup/d00_qc/sub-S2_multiverse_branch_autoreject_check_qc.json "
            "(branch-check script not preserved)",
        },
    ]


def overground_exclusions() -> list[dict]:
    """Return sub-019 (truncated at source) and the duplicates dropped from the primary cohort."""
    qc = "results/pipeline/jacobsen/qc"
    rows = [
        {
            "subject": "019",
            "stage": "all",
            "reason": "raw .fdt truncated at the source (OpenNeuro)",
            "numbers": "2142427 of 24999978 declared samples",
            "source": f"src/config_jacobsen.py; {qc}/jacobsen_dataquality_audit.txt",
        }
    ]
    for subs, reason in (
        ("007", "exact duplicate of sub-005 (identical .fdt and events.tsv)"),
        ("013", "exact duplicate of sub-009 (identical .fdt and events.tsv)"),
        ("006,008", "identical EEG .fdt with different events.tsv (inconsistent pair)"),
    ):
        rows.append(
            {
                "subject": subs,
                "stage": "primary statistic and multiverse reporting (n = 14)",
                "reason": reason,
                "numbers": "kept in the n = 18 reference statistic and in every universe's "
                "t_stats; dropped from dedup_diffs",
                "source": f"{qc}/{DUPLICATE_REPORT}",
            }
        )
    return rows


def exclusions(dirs: dict) -> pd.DataFrame:
    """<dataset>_exclusions.tsv: subject, stage, reason, numbers, source."""
    rows = treadmill_exclusions(dirs) if DATASET == "stepup" else overground_exclusions()
    return pd.DataFrame(rows)


def cohorts() -> dict:
    """<dataset>_cohorts.json: configured, reference, multiverse (and overground) cohorts."""
    ersp_dir = get_dataset_dirs(DATASET)["ersp"]
    out = {
        "dataset": DATASET,
        "label": DATASET_LABELS[DATASET],
        "configured_subjects": SUBJECTS,
        "reference": [s for s in SUBJECTS if (ersp_dir / f"sub-{s}_ersp_beta.npy").exists()],
        "multiverse": MULTIVERSE_SUBJECTS,
    }
    if DATASET == "jacobsen":
        variants = load_cohort_variants(DIR_QC)
        out["primary"] = variants["PRIMARY"]
        out["minimal_sensitivity"] = variants["MINIMAL"]
    for key in [k for k in out if isinstance(out[k], list)]:
        out[f"n_{key}"] = len(out[key])
    return out


def repo_relative(value: object) -> str:
    """Path relative to the repository root (forward slashes); other values as str.

    A path outside the repository (a BETAGAIT_DATASETS_DIR override) is written
    as "<outside repository>" so no local path enters a tracked file.
    """
    if not isinstance(value, Path):
        return str(value)
    try:
        return value.resolve().relative_to(config.DIR_PROJ).as_posix()
    except ValueError:
        return "<outside repository>"


def parameters_used() -> dict:
    """<dataset>_parameters_used.json: every resolved src.config value (paths repo-relative)."""
    snap = {}
    for name in dir(config):
        if name.startswith("_") or name in ("DATASETS",):
            continue
        val = getattr(config, name)
        if isinstance(val, (types.ModuleType, types.FunctionType, type)):
            continue
        try:
            snap[name] = json.loads(json.dumps(val, default=repo_relative))
        except (TypeError, ValueError):
            snap[name] = repr(val)
    return snap
