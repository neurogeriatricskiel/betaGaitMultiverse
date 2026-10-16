"""
prepana02_raw2ica.py -- reference pipeline, stage 2: raw EEG -> preprocessed data + ICA fit.

Purpose
    Load the standing (STAND) and walking (CS) data as one annotated raw
    (filtering, referencing, interpolation and ICA then apply identically
    to both conditions; Makeig et al. 1996, Delorme et al. 2012),
    resample, FIR band-pass, notch, interpolate bad channels, optional ASR,
    average reference; then AutoReject on fixed-length epochs, Extended
    Infomax ICA and ICLabel.
    Treadmill: IC rule src.config.ICLABEL_RULE ("balanced").
    Overground: ICA fitted on a REFERENCE_ICA_FIT_HIGHPASS_HZ high-passed
    copy, eye/muscle rule (src.ica_utils.select_ics_jacobsen_paper_rule).
    Ends by checking the saved files against the configuration
    (src.applied_settings); a mismatch stops the run.
Inputs
    datasets/<ds>/d00_raw/  (BrainVision STAND + CS, or one EEGLAB .set)
Outputs
    datasets/<ds>/d02_prep/sub-*_concat_raw.fif, sub-*_preica_clean_epo.fif,
    sub-*_ica.fif, sub-*_montage.png, sub-*_ica_topos_*.png
    datasets/<ds>/d00_qc/sub-*_preprocessing_qc.json
Manuscript
    Methods 2.3 (preprocessing), 2.4 (ICA).
Run
    BETAGAIT_DATASET=<stepup|jacobsen> python scripts/prepana02_raw2ica.py [subject]
    ASR follows src.config.USE_ASR (treadmill off, overground SD 20 by default).
    One subject per process (optional argument) avoids the slow-down seen
    in long-lived processes on the overground data.
"""

import matplotlib
import matplotlib.pyplot as plt
import mne

from src import config
from src.applied_settings import AppliedSettingsMismatchError, assert_applied_matches
from src.config import ASR_CUTOFF, DATASET, ICLABEL_RULE, REFERENCE_LOWPASS_HZ, USE_ASR
from src.ica_utils import save_ica_component_plots, select_ics_jacobsen_paper_rule
from src.paths import get_dataset_dirs
from src.pipeline_steps import fit_ica, load_and_concatenate, preprocess_raw
from src.qc import log_qc, preprocessing_flag
from src.resume import stage_already_done
from src.stage_runner import run_for_subjects, subjects_from_command_line

# Overground only: ICA fitted on a separately high-passed copy; weights are
# applied back to the analysis data. None for treadmill.
ICA_FIT_HIGHPASS = getattr(config, "REFERENCE_ICA_FIT_HIGHPASS_HZ", None)

# Overground only: the source study's ICLabel rule, P(eye) > 0.9 or
# P(muscle) > 0.9 (the same operation as the multiverse "liberal" rule).
ARTIFACT_RULE_FN = select_ics_jacobsen_paper_rule if DATASET == "jacobsen" else None


def filter_settings() -> tuple[float, float]:
    """Return the (high-pass, low-pass) in Hz that stage 2 applies for the active dataset."""
    return config.REFERENCE_HIGHPASS_HZ, REFERENCE_LOWPASS_HZ


def save_montage_plot(raw: mne.io.BaseRaw, prep_dir, subject: str) -> None:
    """Save the sensor-position plot of the preprocessed data (warn on failure)."""
    try:
        title = f"sub-{subject} -- {config.MONTAGE_SOURCE} montage"
        fig = raw.plot_sensors(show_names=True, title=title, show=False)
        montage_path = prep_dir / f"sub-{subject}_montage.png"
        fig.savefig(montage_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Montage saved -> {montage_path.name}")
    except Exception as e:
        print(f"  [WARN] Montage save failed: {e}")


def log_preprocessing_qc(raw, ica, n_brain_ics: int, dirs: dict, subject: str) -> None:
    """Write the stage-2 QC record (settings applied and ICA counts)."""
    highpass_hz, lowpass_hz = filter_settings()
    flag = preprocessing_flag(n_brain_ics, ica.n_components_)
    rule = ICLABEL_RULE if ARTIFACT_RULE_FN is None else "jacobsen_paper_eye_muscle"
    metrics = {
        "concat_dur_s": round(float(raw.times[-1]), 1),
        "n_ica_comps": ica.n_components_,
        "n_brain_ics": n_brain_ics,
        "n_excl_ics": len(ica.exclude),
        "use_asr": bool(USE_ASR),
        "asr_cutoff": float(ASR_CUTOFF) if USE_ASR else None,
        "highpass_hz": float(highpass_hz),
        "lowpass_hz": float(lowpass_hz),
        "ica_fit_highpass_hz": ICA_FIT_HIGHPASS,
        "iclabel_rule": rule,
    }
    log_qc(qc_dir=dirs["qc"], subject=subject, stage="preprocessing", flag=flag, metrics=metrics)
    print(
        f"  QC preprocessing: {flag}  dur={raw.times[-1]:.0f}s  ica_comps={ica.n_components_}  "
        f"brain_ics={n_brain_ics}  excl={len(ica.exclude)}"
    )


def process_subject(subject: str, dirs: dict) -> None:
    """Preprocess one subject, fit ICA, write QC and check the applied settings."""
    concat_out = dirs["prep"] / f"sub-{subject}_concat_raw.fif"
    ica_path = dirs["prep"] / f"sub-{subject}_ica.fif"
    epo_path = dirs["prep"] / f"sub-{subject}_preica_clean_epo.fif"
    # Outputs must be newer than concat_out (src.resume.stage_already_done).
    if stage_already_done(
        [ica_path, epo_path],
        inputs=[concat_out],
        validate=lambda: mne.preprocessing.read_ica(ica_path, verbose=False),
    ):
        print(f"  Already complete -- skipping sub-{subject}")
        return
    highpass_hz, lowpass_hz = filter_settings()
    raw = preprocess_raw(
        load_and_concatenate(subject, dirs["raw"]),
        subject,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        use_asr=USE_ASR,
        asr_cutoff=ASR_CUTOFF,
        use_gedai=False,
    )
    # Saved as float32 (MNE default); later stages reload it (docs/decisions.md).
    raw.save(concat_out, overwrite=True)
    print(f"  Saved concat raw -> {concat_out.name}")
    save_montage_plot(raw, dirs["prep"], subject)
    # AutoReject -> Extended Infomax -> ICLabel; ICA and epochs saved for prepana03.
    raw, ica, n_brain_ics = fit_ica(
        raw,
        subject,
        iclabel_rule=ICLABEL_RULE,
        ica_path=ica_path,
        iclean_path=None,
        epo_path=epo_path,
        ica_fit_highpass_hz=ICA_FIT_HIGHPASS,
        artifact_rule_fn=ARTIFACT_RULE_FN,
    )
    save_ica_component_plots(ica, dirs["prep"], subject)
    log_preprocessing_qc(raw, ica, n_brain_ics, dirs, subject)
    assert_applied_matches(subject, dirs)
    print(f"  Applied settings match config ({DATASET}).")


def main() -> None:
    """Run stage 2 for one or all subjects."""
    matplotlib.use("Agg")
    subjects = subjects_from_command_line("Preprocessing + ICA fit (reference stage 2).")
    if USE_ASR != config.REFERENCE_USE_ASR:
        print(
            f"[NOTE] BETAGAIT_USE_ASR overrides the {DATASET} reference ASR setting: "
            f"USE_ASR={USE_ASR} (reference default {config.REFERENCE_USE_ASR})."
        )
    dirs = get_dataset_dirs(DATASET)
    run_for_subjects(
        subjects,
        lambda subject: process_subject(subject, dirs),
        reraise=(AppliedSettingsMismatchError,),
    )


if __name__ == "__main__":
    main()
