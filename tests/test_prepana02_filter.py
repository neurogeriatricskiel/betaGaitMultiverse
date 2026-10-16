"""prepana02 passes each dataset's configured band-pass to the filter (synthetic data)."""

import importlib.util
import os
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

EXPECTED = {"stepup": [(1.0, None), (None, 60.0)], "jacobsen": [(0.2, None), (None, 60.0)]}


class _StopError(Exception):
    pass


def _load_prepana02(dataset):
    os.environ["BETAGAIT_DATASET"] = dataset
    os.environ.pop("BETAGAIT_USE_ASR", None)
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        f"prepana02_{dataset}", Path(__file__).parents[1] / "scripts" / "prepana02_raw2ica.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_raw(subject, raw_dir):
    sfreq = 250.0
    info = mne.create_info(["Cz", "C3", "C4", "Pz"], sfreq, "eeg")
    data = np.random.default_rng(0).normal(0, 1e-5, (4, int(120 * sfreq)))
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1005")
    raw.annotations.append(0, 60, "STAND")
    raw.annotations.append(60, 59, "CS")
    return raw


@pytest.mark.parametrize("dataset", ["stepup", "jacobsen"])
def test_filter_arguments(dataset, monkeypatch, tmp_path):
    p02 = _load_prepana02(dataset)
    assert p02.filter_settings() == (EXPECTED[dataset][0][0], EXPECTED[dataset][1][1])

    calls = []
    original = mne.io.BaseRaw.filter

    def spy(self, l_freq, h_freq, *args, **kwargs):
        calls.append((l_freq, h_freq))
        return original(self, l_freq, h_freq, *args, **kwargs)

    def stop(*args, **kwargs):
        raise _StopError

    monkeypatch.setattr(mne.io.BaseRaw, "filter", spy)
    monkeypatch.setattr(p02, "load_and_concatenate", _synthetic_raw)
    monkeypatch.setattr(p02, "fit_ica", stop)
    monkeypatch.setattr(p02, "USE_ASR", False)
    with pytest.raises(_StopError):
        p02.process_subject("synthetic", {"raw": tmp_path, "prep": tmp_path, "qc": tmp_path})
    assert calls == EXPECTED[dataset]
    saved = mne.io.read_raw_fif(tmp_path / "sub-synthetic_concat_raw.fif", verbose=False)
    assert saved.info["highpass"] == pytest.approx(EXPECTED[dataset][0][0])
    assert saved.info["lowpass"] == pytest.approx(60.0)
