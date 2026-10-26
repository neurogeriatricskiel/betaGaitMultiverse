"""Analysis parameters for betaGaitMultiverse: every value the analysis uses.

Dataset selection
-----------------
The active dataset is chosen by the BETAGAIT_DATASET environment variable
("stepup" = treadmill, "jacobsen" = overground; default "stepup"). Every
attribute of the matching per-dataset module (src/config_stepup.py or
src/config_jacobsen.py) is re-exported from here, so scripts import
everything from `src.config` and never branch on the dataset to find a
value. Per-dataset values (subjects, cohorts, file formats, event labels,
reference-pipeline overrides) live in those modules; values shared by
both datasets live below.

Reading this file
-----------------
Sections follow the order of the Methods. Each constant gives its unit
and, in brackets, the manuscript section it belongs to (M2.x = Methods).
"Reference" = the fixed pipeline prepana01-07; "multiverse" = the
27-universe analysis mulana01-04.

Print every value for both datasets side by side with

    python -m src.config
"""

import os
from pathlib import Path

from src import config_jacobsen, config_stepup

# =====================================================================
# 1. Dataset selection and paths                                [M2.1]
# =====================================================================
ACTIVE_DATASET = os.environ.get("BETAGAIT_DATASET", "stepup")

_PER_DATASET = {
    "stepup": config_stepup,
    "jacobsen": config_jacobsen,
}

if ACTIVE_DATASET not in _PER_DATASET:
    raise ValueError(
        f"Unknown ACTIVE_DATASET {ACTIVE_DATASET!r} -- expected one of {sorted(_PER_DATASET)}"
    )

# Display label of each dataset key, used by every figure, QC table and doc.
DATASET_LABELS = {"stepup": "treadmill", "jacobsen": "overground"}

# =====================================================================
# 2. Signal: band-pass, notch, resampling                        [M2.3]
# =====================================================================
TARGET_SFREQ = 250  # Hz -- both datasets resampled to this rate
LINE_FREQ = 50.0  # Hz -- notch filter (EU mains)
# Reference pipeline (prepana02). FIR (firwin), applied as a separate
# high-pass then low-pass by src.pipeline_steps.preprocess_raw. The
# high-pass is per dataset: REFERENCE_HIGHPASS_HZ in config_stepup (1.0 Hz)
# and config_jacobsen (0.2 Hz, the source study's value).
REFERENCE_LOWPASS_HZ = 60.0  # Hz -- low-pass, both datasets
BAD_CHANNEL_ZSCORE = 3.0  # z of per-channel peak-to-peak amplitude -> interpolated
MONTAGE_HEAD_RADIUS_M = 0.095  # m -- scale of overground unit-sphere electrode positions

# =====================================================================
# 3. ASR (Artifact Subspace Reconstruction, asrpy)               [M2.3]
# =====================================================================
# Applied after bad-channel interpolation, before the average reference,
# calibrated on the standing (STAND) segment. Burst correction
# (asrpy ASR.transform). Mullen et al. 2015; Gorjan et al. 2022.
#
# USE_ASR (the reference pipeline's switch) is resolved below, after the
# per-dataset values are loaded: each dataset's REFERENCE_USE_ASR
# (treadmill off, overground SD 20) unless BETAGAIT_USE_ASR overrides it.
ASR_CUTOFF = 20.0  # SD -- reference cutoff, = multiverse "sd20"
ASR_CUTOFF_BY_MODE = {"sd3": 3.0, "sd20": ASR_CUTOFF}  # SD -- multiverse asr_mode -> cutoff
ASR_WIN_LEN = 0.5  # s -- calibration window length
ASR_WIN_OVERLAP = 0.66  # fraction -- calibration window overlap
ASR_METHOD = "euclid"  # covariance estimator
ASR_MEM_SPLITS = 30  # transform() chunking -- memory only, output identical
ASR_CALIBRATION_FLOOR_S = 115.0  # s -- minimum calibration length after trim
ASR_EDGE_TRIM_S = 2.0  # s -- trimmed off the end of the calibration window

# =====================================================================
# 4. ICA and AutoReject                                          [M2.4]
# =====================================================================
ICA_METHOD = "infomax"  # with ICA_FIT_PARAMS -> Extended Infomax
ICA_FIT_PARAMS = {"extended": True}
N_COMPONENTS = 0.99  # fraction of variance retained by the ICA
ICA_DECIM = 2  # ICA.fit(decim=...)
RANDOM_STATE = 42  # seed, AutoReject and ICA, both pipelines
EPOCH_DUR = 2.0  # s -- fixed-length epochs for AutoReject + ICA fit
AUTOREJECT_N_INTERPOLATE = [1, 2, 4]  # AutoReject n_interpolate grid (other args: library defaults)
MIN_CLEAN_EPOCHS = 20  # epochs -- fewer clean epochs after AutoReject = subject fails

# =====================================================================
# 5. IC-selection rules (ICLabel)                                [M2.4]
# =====================================================================
# src.ica_utils.select_ics_by_rule:
#   conservative -> keep IC if P(brain) > ICLABEL_P_BRAIN_CONSERVATIVE
#   balanced     -> keep IC if P(brain) > ICLABEL_P_BRAIN_BALANCED
#   liberal      -> reject IC if P(muscle) or P(eye) > ICLABEL_P_ARTIFACT_LIBERAL
ICLABEL_P_BRAIN_CONSERVATIVE = 0.9
ICLABEL_P_BRAIN_BALANCED = 0.7
ICLABEL_P_ARTIFACT_LIBERAL = 0.9
ICLABEL_RULE = (
    "balanced"  # treadmill reference rule (overground reference: eye/muscle rule = liberal)
)

# =====================================================================
# 6. Epoching and amplitude rejection                       [M2.4, M2.5]
# =====================================================================
AMP_THRESH = 350e-6  # V -- gait cycles and baseline epochs with |x| above this are rejected
BASELINE_EPOCH_DUR_S = 2.0  # s -- fixed-length epochs for the standing-baseline power

# =====================================================================
# 7. Gait cycles, warp, anchors                                  [M2.2, M2.5]
# =====================================================================
# Treadmill events are detected from motion capture (src.gait_cycles):
GAIT_LOWPASS_HZ = 6.0  # Hz -- low-pass of the heel-minus-pelvis marker signal
GAIT_MIN_STEP_S = 0.5  # s -- minimum distance between detected events
GAIT_MIN_STRIDE_S = 0.8  # s -- shorter RHS-RHS intervals are false heel strikes
# Cycle validation, both datasets (src.gait_cycles.extract_valid_gait_cycles):
CYCLE_MIN_DUR_S = 0.8  # s
CYCLE_MAX_DUR_S = 2.0  # s
CYCLE_LTO_MAX_FRAC = 0.30  # fraction of cycle -- LTO later than this = invalid cycle
MIN_GAIT_CYCLES = 20  # cycles -- fewer accepted cycles = subject fails (multiverse)
# Warp: each cycle's power is warped piecewise-linearly so LTO/LHS/RTO
# land on group anchors (% of cycle). Anchors: frozen, data-derived
# group medians in d05_ersp/group_gait_event_anchors_frozen.json
# (src.ersp.load_group_anchors), used by both multiverses and the
# treadmill reference; the overground reference uses the fixed
# config_jacobsen.REFERENCE_WARP_ANCHORS_PCT instead.

# =====================================================================
# 8. TFR and baseline                                            [M2.5]
# =====================================================================
TFR_FMIN = 8.0  # Hz -- TFR floor, both pipelines
PIPELINE_TFR_FMAX = 60.0  # Hz -- reference TFR ceiling (8-60 Hz)
MULTIVERSE_TFR_FMAX = 40.0  # Hz -- multiverse TFR ceiling (8-40 Hz)
TFR_N_CYCLES_DIVISOR = 2.0  # Morlet n_cycles = f / this
TFR_N_POINTS = 101  # gait-cycle grid, 0-100 % in 1 % steps
TFR_EDGE_CROP = 0.05  # fraction cropped at each edge after the TFR
BASELINE_EDGE_TRIM_S = 2.0  # s -- trimmed off the end of the standing baseline
# ERSP = mean over cycles of 10*log10(P_cycle / P_standing), dB.

# =====================================================================
# 9. ROI and beta band                                           [M2.6]
# =====================================================================
ROI_CENTER_CH = "Cz"  # linear ROI: w_i = 1 - d_i / d_max, normalised to sum 1
BETA_FMIN = 13.0  # Hz
BETA_FMAX = 30.0  # Hz

# =====================================================================
# 10. Multiverse: grid, fixed settings, cohorts                  [M2.7]
# =====================================================================
MULTIVERSE_NAME = "beta_gait_multiverse"
# Decision nodes in Table 2.1 order (3 x 3 x 3 = 27 universes).
MULTIVERSE_GRID = {
    "highpass_hz": [0.5, 1.0, 2.0],
    "asr_mode": ["off", "sd3", "sd20"],
    "iclabel_rule": ["conservative", "balanced", "liberal"],
}
MULTIVERSE_LOWPASS_HZ = 60.0  # Hz -- fixed, not a decision node
# Display labels of the decision nodes and their levels (figures, tables).
DECISION_LABELS = {
    "highpass_hz": "High-pass (Hz)",
    "asr_mode": "ASR",
    "iclabel_rule": "IC selection",
}
OPTION_LABELS = {
    "highpass_hz": {0.5: "0.5", 1.0: "1.0", 2.0: "2.0"},
    "asr_mode": {"off": "off", "sd3": "SD 3", "sd20": "SD 20"},
    "iclabel_rule": {
        "conservative": "conservative",
        "balanced": "balanced",
        "liberal": "liberal",
    },
}
# Cohorts: SUBJECTS (reference) and MULTIVERSE_SUBJECTS live in the
# per-dataset modules; the overground primary cohort (n = 14) is read
# from results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt by
# src.qc.load_primary_cohort.

# GEDAI, evaluated as a candidate node and excluded (scripts/qc/qc_gedai_gate.py).
# Gate: GEDAI is viable only if it removes at most GEDAI_GATE_THRESHOLD_PCT of
# the linear Cz-ROI 13-30 Hz power (Welch PSD of the whole STAND + CS recording,
# mean over GEDAI_GATE_SUBJECTS). The 15 % criterion is ours, set before the
# check; it is not taken from the literature.
GEDAI_GATE_THRESHOLD_PCT = 15.0
GEDAI_GATE_SUBJECTS = ("S1", "S9", "S14")  # treadmill
# Settings passed to src.nodes.gedai_node.apply_gedai_node. The broadband
# multiplier is the value src.pipeline_steps.preprocess_raw passes
# (gedai_noise_multiplier = 3.0); the node's own default is 6.0, run as a
# sensitivity check (GEDAI_SENSITIVITY_BROADBAND_MULTIPLIER).
GEDAI_SETTINGS = {
    "duration": 2.0,
    "overlap": 0.5,
    "broadband_noise_multiplier": 3.0,
    "spectral_noise_multiplier": 3.0,
    "wavelet_type": "haar",
    "wavelet_level": 5,
    "wavelet_low_cutoff": 2.0,
    "sensai_method": "optimize",
}
GEDAI_SENSITIVITY_BROADBAND_MULTIPLIER = 6.0

# =====================================================================
# 11. Statistics                                                 [M2.8]
# =====================================================================
# Per subject: double-stance minus swing beta ERSP (ROI, dB). Group:
# one-sample t-test of these differences against 0 (computed as the
# equivalent paired t-test, scipy.stats.ttest_rel), two-sided.
STATS_TEST = "one-sample t-test of per-subject double-stance - swing vs 0 (two-sided)"
ALPHA = 0.05
MULTIPLE_COMPARISON_CORRECTION = None  # universes are reported uncorrected

# =====================================================================
# Re-export the active dataset's values
# =====================================================================
_active = _PER_DATASET[ACTIVE_DATASET]
for _name in dir(_active):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_active, _name)
del _name

# Reference ASR switch: per-dataset default, explicit env override only.
_asr_env = os.environ.get("BETAGAIT_USE_ASR")
USE_ASR = (
    _active.REFERENCE_USE_ASR if _asr_env is None else _asr_env.lower() not in ("false", "0", "no")
)


# =====================================================================
# Directories (created on import)
# =====================================================================
def define_dir(root: Path, *names: str) -> Path:
    """Create a directory (parents included) and return it as a Path."""
    path = root
    for name in names:
        path = path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


DIR_PROJ = Path(__file__).resolve().parents[1]
# BETAGAIT_DATASETS_DIR redirects all per-dataset data (inputs and outputs) to
# another folder, e.g. to rerun a stage for one subject without touching datasets/.
_datasets_env = os.environ.get("BETAGAIT_DATASETS_DIR")
DIR_DATASETS = (
    define_dir(Path(_datasets_env)) if _datasets_env else define_dir(DIR_PROJ, "datasets")
)
DIR_SCRIPTS = define_dir(DIR_PROJ, "scripts")
DIR_RESULTS = define_dir(DIR_PROJ, "results")

# Per-dataset group statistics and QC tables.
DIR_QC = define_dir(DIR_RESULTS, "pipeline", ACTIVE_DATASET, "qc")

# Final figures for both datasets (file names carry the dataset key).
DIR_FIGURES = define_dir(DIR_RESULTS, "figures")
DIR_SUPPLEMENTARY = define_dir(DIR_RESULTS, "supplementary")

# Multiverse: branch ICA cache, COMET working directory (universe scripts
# in comet/scripts/, combined results in comet/results/), outputs.
DIR_MULTIVERSE = define_dir(DIR_RESULTS, "multiverse", ACTIVE_DATASET)
DIR_MULTIVERSE_OUTPUTS = define_dir(DIR_MULTIVERSE, "outputs")
DIR_MULTIVERSE_BRANCHES = define_dir(DIR_MULTIVERSE, "branches")
DIR_MULTIVERSE_COMET = define_dir(DIR_MULTIVERSE, "comet")

# Archived treadmill multiverse run with a 40 Hz low-pass (read-only; supplement).
DIR_MULTIVERSE_LOWPASS40_ARCHIVE = DIR_RESULTS / "multiverse" / "stepup_40hz_archive"
ARCHIVE_LOWPASS_HZ = 40.0  # Hz -- low-pass of the archived treadmill run [M2.7]

# Per-dataset data trees for src.paths.get_dataset_dirs(), built for both
# datasets regardless of ACTIVE_DATASET.
DATASETS = {
    dataset_name: {
        "root": define_dir(DIR_DATASETS, mod.ROOT_DIRNAME),
        "dirs": mod.DIRS,
    }
    for dataset_name, mod in _PER_DATASET.items()
}


# =====================================================================
# python -m src.config : parameter table for both datasets
# =====================================================================
def parameter_rows() -> list[tuple]:
    """(parameter, treadmill value, overground value, section) rows."""
    st, jc = config_stepup, config_jacobsen

    def both(value):
        return (value, value)

    anchors_note = "frozen group medians (load_group_anchors)"
    return [
        (
            "Subjects configured (reference)",
            f"n = {len(st.SUBJECTS)} (S21 fails -> 17)",
            f"n = {len(jc.SUBJECTS)} (primary n = 14)",
            "M2.1",
        ),
        (
            "Subjects (multiverse)",
            f"n = {len(st.MULTIVERSE_SUBJECTS)}",
            f"n = {len(jc.MULTIVERSE_SUBJECTS)} (primary n = 14)",
            "M2.1",
        ),
        ("EEG format", st.EEG_FORMAT, jc.EEG_FORMAT, "M2.1"),
        ("Montage", st.MONTAGE_SOURCE, jc.MONTAGE_SOURCE, "M2.1"),
        ("Gait events", st.EVENT_SOURCE, jc.EVENT_SOURCE, "M2.2"),
        ("Resample (Hz)", *both(TARGET_SFREQ), "M2.3"),
        (
            "High-pass, reference (Hz, FIR)",
            st.REFERENCE_HIGHPASS_HZ,
            jc.REFERENCE_HIGHPASS_HZ,
            "M2.3",
        ),
        ("Low-pass, reference (Hz, FIR)", *both(REFERENCE_LOWPASS_HZ), "M2.3"),
        ("Notch (Hz)", *both(LINE_FREQ), "M2.3"),
        ("Bad channels: |z| peak-to-peak >", *both(BAD_CHANNEL_ZSCORE), "M2.3"),
        (
            "ASR, reference",
            f"SD {ASR_CUTOFF:g}" if st.REFERENCE_USE_ASR else "off",
            f"SD {ASR_CUTOFF:g}, burst correction" if jc.REFERENCE_USE_ASR else "off",
            "M2.3",
        ),
        ("ASR calibration", *both(f"STAND, end trimmed {ASR_EDGE_TRIM_S:g} s"), "M2.3"),
        (
            "ICA",
            *both(f"Extended Infomax, {N_COMPONENTS} variance, seed {RANDOM_STATE}"),
            "M2.4",
        ),
        (
            "ICA fitted on",
            "analysis data",
            f"{jc.REFERENCE_ICA_FIT_HIGHPASS_HZ:g} Hz high-passed copy",
            "M2.4",
        ),
        (
            "AutoReject",
            *both(
                f"{EPOCH_DUR:g} s epochs, n_interpolate {AUTOREJECT_N_INTERPOLATE}, "
                f"seed {RANDOM_STATE}, min {MIN_CLEAN_EPOCHS} clean"
            ),
            "M2.4",
        ),
        (
            "IC rule, reference",
            ICLABEL_RULE,
            f"eye/muscle P > {ICLABEL_P_ARTIFACT_LIBERAL} (= liberal)",
            "M2.4",
        ),
        ("Amplitude rejection (uV)", *both(f"{AMP_THRESH * 1e6:g}"), "M2.5"),
        ("Baseline", "STAND recording", "start_standing - end_standing", "M2.5"),
        ("Baseline end trim (s)", *both(BASELINE_EDGE_TRIM_S), "M2.5"),
        ("Cycle duration (s)", *both(f"{CYCLE_MIN_DUR_S:g}-{CYCLE_MAX_DUR_S:g}"), "M2.2"),
        (
            "Warp anchors, reference",
            anchors_note,
            f"fixed {'/'.join(f'{a:g}' for a in jc.REFERENCE_WARP_ANCHORS_PCT)} %",
            "M2.5",
        ),
        ("Warp anchors, multiverse", *both(anchors_note), "M2.5"),
        (
            "TFR",
            *both(
                f"Morlet, n_cycles = f/{TFR_N_CYCLES_DIVISOR:g}, "
                f"{TFR_EDGE_CROP:.0%} edge crop, {TFR_N_POINTS} points"
            ),
            "M2.5",
        ),
        ("TFR range, reference (Hz)", *both(f"{TFR_FMIN:g}-{PIPELINE_TFR_FMAX:g}"), "M2.5"),
        ("TFR range, multiverse (Hz)", *both(f"{TFR_FMIN:g}-{MULTIVERSE_TFR_FMAX:g}"), "M2.5"),
        ("ROI", *both(f"linear, centred on {ROI_CENTER_CH}"), "M2.6"),
        ("Beta band (Hz)", *both(f"{BETA_FMIN:g}-{BETA_FMAX:g}"), "M2.6"),
        ("Statistics", *both(f"{STATS_TEST}, alpha {ALPHA}, uncorrected"), "M2.8"),
    ]


def grid_rows() -> list[tuple[str, str]]:
    """Return the (decision, levels) rows of the multiverse grid and its fixed settings."""
    rows = [
        (DECISION_LABELS[node], ", ".join(OPTION_LABELS[node][lv] for lv in levels))
        for node, levels in MULTIVERSE_GRID.items()
    ]
    return rows + [
        ("Low-pass (Hz)", f"{MULTIVERSE_LOWPASS_HZ:g} (fixed)"),
        ("ASR cutoffs", str(ASR_CUTOFF_BY_MODE)),
    ]


def print_parameter_table() -> None:
    """Print every parameter for both datasets side by side (python -m src.config)."""
    rows = parameter_rows()
    headers = ("Parameter", DATASET_LABELS["stepup"], DATASET_LABELS["jacobsen"], "Section")
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(4)]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    print(line.format(*headers))
    print(line.format(*("-" * w for w in widths)))
    for r in rows:
        print(line.format(*(str(v) for v in r)))
    print("\nMultiverse grid (both datasets, 27 universes):")
    for label, levels in grid_rows():
        print(f"  {label:<16} {levels}")


def markdown_parameter_table() -> str:
    """Return the parameter table and the multiverse grid as Markdown (docs/parameters.md)."""
    headers = ("Parameter", DATASET_LABELS["stepup"], DATASET_LABELS["jacobsen"], "Section")
    lines = [
        "# Parameters",
        "",
        "Generated by `python -m src.config --markdown > docs/parameters.md`; do not edit by hand.",
        "Values come from `src/config.py`, `src/config_stepup.py` (treadmill) and",
        "`src/config_jacobsen.py` (overground). Section = Methods section (M2.x).",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    lines += [
        "| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in parameter_rows()
    ]
    lines += [
        "",
        "## Multiverse grid (both datasets, 27 universes)",
        "",
        "| Decision | Levels |",
        "|---|---|",
    ]
    lines += [f"| {label} | {levels} |" for label, levels in grid_rows()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    if "--markdown" in sys.argv[1:]:
        sys.stdout.write(markdown_parameter_table())
    else:
        print_parameter_table()
