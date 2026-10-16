"""Manuscript parameters (Methods) as resolved by src.config, for both datasets."""

import importlib
import os
import sys

import pytest


def load_config(dataset):
    os.environ["BETAGAIT_DATASET"] = dataset
    os.environ.pop("BETAGAIT_USE_ASR", None)
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    return importlib.import_module("src.config")


@pytest.fixture(params=["stepup", "jacobsen"])
def cfg(request):
    return load_config(request.param)


def test_both_datasets_resolve(cfg):
    assert cfg.DATASET == cfg.ACTIVE_DATASET
    assert cfg.DATASET_LABELS == {"stepup": "treadmill", "jacobsen": "overground"}


def test_signal_parameters(cfg):
    assert cfg.TARGET_SFREQ == 250
    assert cfg.LINE_FREQ == 50.0
    assert cfg.REFERENCE_HIGHPASS_HZ == {"stepup": 1.0, "jacobsen": 0.2}[cfg.DATASET]
    assert cfg.REFERENCE_LOWPASS_HZ == 60.0
    assert cfg.BAD_CHANNEL_ZSCORE == 3.0


def test_asr(cfg):
    assert cfg.ASR_CUTOFF == 20.0
    assert cfg.ASR_CUTOFF_BY_MODE == {"sd3": 3.0, "sd20": 20.0}
    expected_as_run = {"stepup": False, "jacobsen": True}[cfg.DATASET]
    assert cfg.REFERENCE_USE_ASR is expected_as_run
    assert cfg.USE_ASR is expected_as_run  # default, no BETAGAIT_USE_ASR


def test_ica_and_autoreject(cfg):
    assert cfg.ICA_METHOD == "infomax" and cfg.ICA_FIT_PARAMS == {"extended": True}
    assert cfg.N_COMPONENTS == 0.99
    assert cfg.RANDOM_STATE == 42
    assert cfg.EPOCH_DUR == 2.0
    assert cfg.AUTOREJECT_N_INTERPOLATE == [1, 2, 4]
    assert cfg.MIN_CLEAN_EPOCHS == 20


def test_ic_rules(cfg):
    assert cfg.ICLABEL_P_BRAIN_CONSERVATIVE == 0.9
    assert cfg.ICLABEL_P_BRAIN_BALANCED == 0.7
    assert cfg.ICLABEL_P_ARTIFACT_LIBERAL == 0.9
    assert cfg.ICLABEL_RULE == "balanced"


def test_overground_reference_overrides():
    cfg = load_config("jacobsen")
    assert cfg.REFERENCE_ICA_FIT_HIGHPASS_HZ == 2.0
    assert cfg.REFERENCE_WARP_ANCHORS_PCT == (18.0, 50.0, 68.0)


def test_tfr_baseline_roi(cfg):
    assert cfg.AMP_THRESH == 350e-6
    assert cfg.TFR_N_CYCLES_DIVISOR == 2.0
    assert cfg.TFR_EDGE_CROP == 0.05
    assert cfg.TFR_N_POINTS == 101
    assert (cfg.TFR_FMIN, cfg.PIPELINE_TFR_FMAX, cfg.MULTIVERSE_TFR_FMAX) == (8.0, 60.0, 40.0)
    assert (cfg.BETA_FMIN, cfg.BETA_FMAX) == (13.0, 30.0)
    assert cfg.BASELINE_EDGE_TRIM_S == 2.0
    assert cfg.ROI_CENTER_CH == "Cz"


def test_multiverse_grid_and_statistics(cfg):
    assert cfg.MULTIVERSE_GRID == {
        "highpass_hz": [0.5, 1.0, 2.0],
        "asr_mode": ["off", "sd3", "sd20"],
        "iclabel_rule": ["conservative", "balanced", "liberal"],
    }
    assert cfg.MULTIVERSE_LOWPASS_HZ == 60.0
    assert cfg.ALPHA == 0.05 and cfg.MULTIPLE_COMPARISON_CORRECTION is None


def test_cohort_sizes():
    st = load_config("stepup")
    assert (len(st.SUBJECTS), len(st.MULTIVERSE_SUBJECTS)) == (18, 15)
    assert {"S2", "S10", "S21"} == set(st.SUBJECTS) - set(st.MULTIVERSE_SUBJECTS)
    jc = load_config("jacobsen")
    assert len(jc.SUBJECTS) == len(jc.MULTIVERSE_SUBJECTS) == 18


def test_parameter_table_rows():
    cfg = load_config("stepup")
    rows = cfg.parameter_rows()
    assert all(len(r) == 4 for r in rows)
    assert any(r[0].startswith("ASR, reference") and r[1] == "off" for r in rows)
