"""Per-dataset settings of the treadmill dataset (key "stepup").

stepUpAms, VU Amsterdam treadmill walking, BrainVision EEG + optical
motion capture.

Pure data (no filesystem access, no imports from src.config), re-exported
by src.config when BETAGAIT_DATASET=stepup.
"""

DATASET = "stepup"
ROOT_DIRNAME = "stepup"  # datasets/stepup/

# Full cohort.
SUBJECTS = [
    "S1",
    "S2",
    "S3",
    "S4",
    "S7",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S20",
    "S21",
    "S23",
]

# Subjects for multiverse analysis.
# Excluded (verified 2026-10-01 -- see each subject's own QC source below;
# results/multiverse/stepup/branches/{S2,S10,S21}/ have 0 cached ICA
# branches for all three -- none was ever run in the multiverse that
# produced the current multiverse_results.pkl):
#   S10 — ICA failure: 1/38 brain ICs (brain_frac=0.026), signal quality
#         insufficient. Source: datasets/stepup/d00_qc/sub-S10_ica_qc.json.
#   S21 — AutoReject failure under the current ASR-off reference config:
#         only 12/243 clean fixed-length (2.0s) epochs (reject_frac=0.951),
#         below fit_ica's own 20-epoch floor. Reproduced directly from
#         datasets/stepup/d02_prep/sub-S21_concat_raw.fif (the current
#         ASR-off preprocessed raw) -- source:
#         datasets/stepup/d00_qc/sub-S21_autoreject_recheck_qc.json.
#   S2  — NOT a blanket AutoReject failure (the "12/243" previously listed
#         here was S21's number, copied in error -- S2 passes AutoReject
#         under the reference-pipeline-equivalent ASR-off/HP=1.0/LP=60.0
#         setting with 40/243 clean epochs, which is why S2's reference-
#         pipeline run completed in full, unlike S21's). Re-checked
#         directly from raw EEG across all 9 of the multiverse's unique
#         ICA branches (3 highpass x 3 asr_mode, lowpass=60.0 fixed): S2
#         fails (<20 clean epochs) ONLY under asr_mode="sd20" at
#         highpass=0.5 (6/243 clean) and highpass=1.0 (12/243 clean) --
#         i.e. 2 of 9 branches / 6 of 27 universes (the 3 iclabel_rule
#         variants of each failing branch); it passes under asr_mode="off"
#         and "sd3" at every highpass, and under asr_mode="sd20" at
#         highpass=2.0 (243/243 clean). Excluded as a single subject-level
#         decision for simplicity, not because it fails universally.
#         Source: datasets/stepup/d00_qc/sub-S2_multiverse_branch_autoreject_check_qc.json.
MULTIVERSE_SUBJECTS = [
    "S1",
    "S3",
    "S4",
    "S7",
    "S9",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S20",
    "S23",
]

# --- Recording / file format ---
EEG_FORMAT = "brainvision"  # .vhdr/.eeg/.vmrk, loaded via mne.io.read_raw_brainvision
TASK_STAND = "STAND"  # BIDS task suffix for the quiet-standing recording
TASK_WALK = "CS"  # BIDS task suffix for the walking recording
MONTAGE_SOURCE = "standard_1005"  # assigned programmatically; no electrodes.tsv in this dataset

# --- Gait event source ---
EVENT_SOURCE = "mocap"  # prepana01 detects HS/TO from motion-capture heel/pelvis markers
MOCAP_SFREQ = 100  # Hz -- motion-capture sampling rate

# --- Reference pipeline filter ---
REFERENCE_HIGHPASS_HZ = 1.0  # Hz -- FIR high-pass of the reference pipeline (prepana02)

# --- Reference pipeline ASR ---
# Treadmill reference: ASR off (the setting of the stored results; see the
# use_asr field of datasets/stepup/d00_qc/sub-*_preprocessing_qc.json).
# Default of src.config.USE_ASR for this dataset; BETAGAIT_USE_ASR overrides.
REFERENCE_USE_ASR = False

# Annotations set by the BrainVision loader (src.pipeline_steps): "STAND"
# (quiet-standing recording, the ERSP baseline) and "CS" (walking).

# Directory layout, relative to datasets/stepup/
DIRS = {
    "qc": "d00_qc",
    "raw": "d00_raw",
    "gait_events": "d01_gaitevents",
    "prep": "d02_prep",
    "clean": "d03_clean",
    "gaitepochs": "d04_gaitepochs",
    "ersp": "d05_ersp",
    "roi_topo": "d05_roi_topo",
}
