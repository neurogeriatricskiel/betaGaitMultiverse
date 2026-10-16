"""
qc_ica_audit.py -- what the reference ICA removed and what it kept at Fpz.

Purpose
    Per subject, ICLabel is recomputed from the saved stage-2 ICA exactly as
    prepana03 does (src.ica_utils.iclabel_probabilities on the AutoReject
    epochs) and the dataset's rule is applied; the exclusions must equal the
    saved ones. Reports the components removed by dominant ICLabel class,
    retained components with P(eye) or P(muscle) > 0.3, and the 13-30 Hz
    variance at Fpz during walking after ICA relative to before, for the
    retained components alone (no PCA residual) and for the three largest
    single retained components. Treadmill also: the reference statistic
    without S10 (one retained component).
Inputs
    datasets/<ds>/d02_prep/sub-*_ica.fif, _preica_clean_epo.fif, _concat_raw.fif
    datasets/<ds>/d03_clean/sub-*_clean-ica.fif, _desc-icaClean_concat_raw.fif
    datasets/<ds>/d05_ersp/ (cohort, S10 check)
Outputs
    results/pipeline/<ds>/qc/<ds>_ica_audit.txt
Manuscript
    Discussion (frontal channels and ICA).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/qc/qc_ica_audit.py [subject ...]
    (default: reference cohort; about 15 s per treadmill subject, 1 min per overground subject)
"""

import sys

import mne
import numpy as np

from src import roi_diagnostics as rd
from src.config import DATASET, DIR_QC, ICLABEL_RULE
from src.ica_utils import iclabel_probabilities, select_ics_by_rule, select_ics_jacobsen_paper_rule
from src.paths import get_dataset_dirs
from src.reference_stats import paired_test, phase_beta_values

CLASSES = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]
RULE = "jacobsen paper eye/muscle > 0.9" if DATASET == "jacobsen" else ICLABEL_RULE


def excluded(probs: np.ndarray) -> list[int]:
    """Components the reference pipeline removes for the active dataset."""
    if DATASET == "jacobsen":
        return select_ics_jacobsen_paper_rule(probs)
    return select_ics_by_rule(probs, ICLABEL_RULE)


def ic_label(probs: np.ndarray, i: int) -> str:
    """Dominant class and the eye / muscle probabilities of one component."""
    k = int(probs[i].argmax())
    return f"{CLASSES[k]} {probs[i, k]:.2f}; eye {probs[i, 2]:.2f}, muscle {probs[i, 1]:.2f}"


def audit_subject(s: str, dirs: dict) -> tuple[list[str], float, float]:
    """Report lines of one subject and its two Fpz variance fractions."""
    ica = mne.preprocessing.read_ica(dirs["prep"] / f"sub-{s}_ica.fif", verbose=False)
    saved = mne.preprocessing.read_ica(dirs["clean"] / f"sub-{s}_clean-ica.fif", verbose=False)
    epo = mne.read_epochs(dirs["prep"] / f"sub-{s}_preica_clean_epo.fif", verbose=False)
    probs, _ = iclabel_probabilities(ica, epo)
    excl = excluded(probs)
    kept = [i for i in range(ica.n_components_) if i not in excl]
    dom = probs.argmax(1)
    by_class = {CLASSES[k]: int(np.sum(dom[excl] == k)) for k in range(7) if np.sum(dom[excl] == k)}
    suspect = [i for i in kept if probs[i, 2] > 0.3 or probs[i, 1] > 0.3]
    raw = mne.io.read_raw_fif(dirs["prep"] / f"sub-{s}_concat_raw.fif", preload=True, verbose=False)
    clean_path = dirs["clean"] / f"sub-{s}_desc-icaClean_concat_raw.fif"
    v_pre = np.var(rd.walking_beta(raw, "Fpz"))
    after = np.var(
        rd.walking_beta(mne.io.read_raw_fif(clean_path, preload=True, verbose=False), "Fpz")
    )
    alone = np.var(rd.walking_beta(rd.retained_projection(raw, ica, kept), "Fpz"))
    single = sorted(
        (
            (np.var(rd.walking_beta(rd.retained_projection(raw, ica, [i]), "Fpz")) / v_pre, i)
            for i in kept
        ),
        reverse=True,
    )
    sus = ", ".join(f"IC{i} eye {probs[i, 2]:.2f} muscle {probs[i, 1]:.2f}" for i in suspect)
    lines = [
        "",
        f"sub-{s}: {ica.n_components_} ICs, removed {len(excl)} {by_class}, kept {len(kept)}; "
        f"recomputed exclusions equal saved: {sorted(excl) == sorted(saved.exclude)}",
        f"  Retained ICs with P(eye) or P(muscle) > 0.3: {sus or 'none'}",
        f"  Fpz beta (13-30 Hz, walking) variance after ICA / before: {100 * after / v_pre:.1f} %; "
        f"retained ICs alone: {100 * alone / v_pre:.1f} %",
        "  Largest single retained ICs at Fpz (% of pre-ICA beta variance): "
        + ", ".join(f"IC{i} {100 * v:.1f} % ({ic_label(probs, i)})" for v, i in single[:3]),
    ]
    return lines, after / v_pre, alone / v_pre


def without_subject_lines(cohort: list[str], dirs: dict, drop: str) -> list[str]:
    """Return the reference paired test with and without one subject."""
    values = {s: phase_beta_values(s, dirs["ersp"]) for s in cohort}
    lines = []
    for label, subs in (
        (f"all n={len(cohort)}", cohort),
        (f"without {drop}", [s for s in cohort if s != drop]),
    ):
        r = paired_test([values[s][0] for s in subs], [values[s][1] for s in subs])
        lines.append(
            f"{label}: t({r['n'] - 1}) = {r['t']:.3f}  p = {r['p']:.4g}  mean diff = "
            f"{r['mean_diff']:+.3f} dB  DS>swing {r['n_ds_gt_swing']}/{r['n']}"
        )
    return lines


def main() -> None:
    """Audit the ICA of the given subjects (default: reference cohort) and write the report."""
    dirs = get_dataset_dirs(DATASET)
    cohort = rd.reference_cohort(DATASET, dirs)
    subjects = sys.argv[1:] or cohort
    lines = [f"=== {DATASET}: ICA audit (rule {RULE})"]
    fractions = []
    for s in subjects:
        sub_lines, after, alone = audit_subject(s, dirs)
        lines += sub_lines
        fractions.append((after, alone))
        print("\n".join(sub_lines), flush=True)
    if len(fractions) > 1:
        a, b = (np.array(x) for x in zip(*fractions, strict=True))
        lines += [
            "",
            f"Across {len(fractions)} subjects: Fpz beta variance kept after ICA median "
            f"{100 * np.median(a):.1f} % (range {100 * a.min():.1f}-{100 * a.max():.1f}); "
            f"retained ICs alone median {100 * np.median(b):.1f} %",
        ]
    if DATASET == "stepup" and "S10" in cohort:
        lines += without_subject_lines(cohort, dirs, "S10")
    (DIR_QC / f"{DATASET}_ica_audit.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
