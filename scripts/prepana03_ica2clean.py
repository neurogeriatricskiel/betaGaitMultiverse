"""
prepana03_ica2clean.py -- reference pipeline, stage 3: ICLabel + ICA application.

Purpose
    Classify the ICA components fitted in stage 2 with ICLabel
    (Pion-Tonachini et al. 2019), mark artefact components with the
    dataset's reference rule (treadmill: "balanced", P(brain) > 0.7;
    overground: P(eye) or P(muscle) > 0.9) and apply the ICA to the
    concatenated STAND + CS raw.
Inputs
    datasets/<ds>/d02_prep/sub-*_concat_raw.fif, sub-*_ica.fif,
    sub-*_preica_clean_epo.fif
Outputs
    datasets/<ds>/d03_clean/sub-*_desc-icaClean_concat_raw.fif, sub-*_clean-ica.fif
    datasets/<ds>/d00_qc/sub-*_ica_qc.json
Manuscript
    Methods 2.4 (ICA and component selection).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/prepana03_ica2clean.py [subject]
"""

import mne
import numpy as np

from src.config import DATASET, ICLABEL_RULE
from src.ica_utils import label_and_mark_ica, select_ics_jacobsen_paper_rule
from src.paths import get_dataset_dirs
from src.pipeline_steps import apply_ica
from src.qc import ica_flag, log_qc
from src.resume import stage_already_done
from src.stage_runner import run_for_subjects, subjects_from_command_line

# Overground only: the source study's eye/muscle rule (see prepana02).
ARTIFACT_RULE_FN = select_ics_jacobsen_paper_rule if DATASET == "jacobsen" else None


def log_ica_qc(ica, n_brain: int, dirs: dict, subject: str) -> None:
    """Write the stage-3 QC record (components kept and excluded)."""
    n_total = len(ica.mixing_matrix_.T)
    brain_frac = n_brain / n_total if n_total > 0 else 0.0
    flag = ica_flag(n_brain, brain_frac)
    metrics = {
        "n_total_comps": n_total,
        "n_brain_comps": n_brain,
        "n_excl_comps": len(ica.exclude),
        "brain_frac": round(brain_frac, 3),
    }
    if n_brain == 0:
        metrics["note"] = "no brain ICs found -- subject skipped"
    log_qc(qc_dir=dirs["qc"], subject=subject, stage="ica", flag=flag, metrics=metrics)
    print(
        f"  QC ica: {flag}  brain={n_brain}/{n_total}  "
        f"excluded={len(ica.exclude)}  brain_frac={brain_frac:.3f}"
    )


def process_subject(subject: str, dirs: dict) -> None:
    """Label the components of one subject, apply the ICA and save the data."""
    ica_in_path = dirs["prep"] / f"sub-{subject}_ica.fif"
    ica_out = dirs["clean"] / f"sub-{subject}_clean-ica.fif"
    raw_out = dirs["clean"] / f"sub-{subject}_desc-icaClean_concat_raw.fif"
    # Stale if prepana02 refitted the ICA after this stage last ran.
    if stage_already_done(
        [ica_out, raw_out],
        inputs=[ica_in_path],
        validate=lambda: mne.io.read_raw_fif(raw_out, preload=False, verbose=False),
    ):
        print(f"  Already complete -- skipping sub-{subject}")
        return
    raw_path = dirs["prep"] / f"sub-{subject}_concat_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    ica = mne.preprocessing.read_ica(ica_in_path, verbose=False)
    epochs_path = dirs["prep"] / f"sub-{subject}_preica_clean_epo.fif"
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    print(f"\n  ICA components : {ica.n_components_}")
    print(f"  Epochs for ICLabel: {len(epochs)} x {len(epochs.ch_names)} ch")
    result = label_and_mark_ica(ica, epochs, rule=ICLABEL_RULE, artifact_rule_fn=ARTIFACT_RULE_FN)
    print(f"\n  Brain ICs kept   : {len(result['brain_ics'])}  {result['brain_ics']}")
    print(f"  Artefact ICs out : {len(result['exclude_ics'])}  {result['exclude_ics']}")
    if len(result["brain_ics"]) == 0:
        print(f"  [FAIL] sub-{subject}: no brain ICs found -- skipping.")
        log_ica_qc(ica, 0, dirs, subject)
        return
    raw_clean = apply_ica(raw, ica, subject, iclean_path=None)  # drops no-position channels
    if np.mean(np.isnan(raw_clean.get_data())) > 0:
        print("  ERROR: NaN in ICA output -- inspect component mixing matrix.")
        return
    ica.save(ica_out, overwrite=True, verbose=False)
    raw_clean.save(raw_out, overwrite=True, verbose=False)
    print(f"  Saved -> {ica_out.name}, {raw_out.name}")
    log_ica_qc(ica, len(result["brain_ics"]), dirs, subject)


def main() -> None:
    """Run stage 3 for one or all subjects."""
    subjects = subjects_from_command_line("ICLabel + ICA application (reference stage 3).")
    dirs = get_dataset_dirs(DATASET)
    run_for_subjects(subjects, lambda subject: process_subject(subject, dirs))


if __name__ == "__main__":
    main()
