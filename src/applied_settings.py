"""
Applied-versus-configured check of the reference preprocessing.

Reads what was actually applied from the saved files of one subject and
compares it with the resolved src.config:
  d02_prep/sub-*_concat_raw.fif  info highpass, lowpass, sfreq; 50 Hz notch
                                 (estimated: Welch PSD at LINE_FREQ vs 2-4 Hz
                                 either side; applied if < NOTCH_MAX_DB)
  d02_prep/sub-*_ica.fif         ICA method, extended flag, variance retained,
                                 high-pass of the data the ICA was fitted on
  d00_qc/sub-*_preprocessing_qc.json  ASR flag and cutoff (not stored in Raw)
Used by scripts/qc/qc_applied_settings.py and at the end of prepana02.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import mne
import numpy as np

from src import config

if TYPE_CHECKING:
    from pathlib import Path


class AppliedSettingsMismatchError(RuntimeError):
    """Saved files do not reflect the configured preprocessing."""


NOTCH_MAX_DB = -3.0  # dB at LINE_FREQ relative to neighbours; un-notched data are positive
NOTCH_SECONDS = 120.0  # s of data used for the notch estimate


def expected_settings() -> dict[str, object]:
    """Return the configured preprocessing values for the active dataset.

    Returns
    -------
    dict of str to object
        Setting name to configured value.
    """
    ica_fit_hp = getattr(config, "REFERENCE_ICA_FIT_HIGHPASS_HZ", None)
    return {
        "highpass_hz": config.REFERENCE_HIGHPASS_HZ,
        "lowpass_hz": config.REFERENCE_LOWPASS_HZ,
        "sfreq_hz": float(config.TARGET_SFREQ),
        "notch_applied": True,
        "use_asr": bool(config.USE_ASR),
        "asr_cutoff": float(config.ASR_CUTOFF) if config.USE_ASR else None,
        "ica_method": "infomax" + (" extended" if config.ICA_FIT_PARAMS.get("extended") else ""),
        "ica_variance": config.N_COMPONENTS,
        "ica_fit_highpass_hz": ica_fit_hp
        if ica_fit_hp is not None
        else config.REFERENCE_HIGHPASS_HZ,
    }


def notch_attenuation_db(raw: mne.io.BaseRaw) -> float:
    """Estimate line-noise attenuation from the median-over-channels Welch PSD.

    Uses the first ``NOTCH_SECONDS`` of data.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to check.

    Returns
    -------
    float
        Power at LINE_FREQ relative to the mean 2-4 Hz either side (dB).
    """
    seg = raw.copy().crop(0, min(NOTCH_SECONDS, raw.times[-1]))
    seg.load_data(verbose=False)
    sfreq = seg.info["sfreq"]
    lf = config.LINE_FREQ
    psd, freqs = mne.time_frequency.psd_array_welch(
        seg.get_data(), sfreq, fmin=lf - 10, fmax=lf + 10, n_fft=int(sfreq * 4), verbose=False
    )
    p = np.median(psd, axis=0)
    dist = np.abs(freqs - lf)
    return float(10 * np.log10(p[np.argmin(dist)] / p[(dist >= 2) & (dist <= 4)].mean()))


def applied_settings(subject: str, dirs: dict[str, Path]) -> dict[str, object]:
    """Read the applied preprocessing values from one subject's saved files.

    Parameters
    ----------
    subject : str
        Subject ID.
    dirs : dict of str to Path
        Dataset directories (uses ``"prep"`` and ``"qc"``).

    Returns
    -------
    dict of str to object
        Setting name to applied value (None if the file is missing).
    """
    out = dict.fromkeys(expected_settings())
    prep = dirs["prep"] / f"sub-{subject}_concat_raw.fif"
    if prep.exists():
        raw = mne.io.read_raw_fif(prep, preload=False, verbose=False)
        out["highpass_hz"] = raw.info["highpass"]
        out["lowpass_hz"] = raw.info["lowpass"]
        out["sfreq_hz"] = raw.info["sfreq"]
        out["notch_db"] = round(notch_attenuation_db(raw), 1)
        out["notch_applied"] = out["notch_db"] < NOTCH_MAX_DB
    ica_path = dirs["prep"] / f"sub-{subject}_ica.fif"
    if ica_path.exists():
        ica = mne.preprocessing.read_ica(ica_path, verbose=False)
        out["ica_method"] = ica.method + (" extended" if ica.fit_params.get("extended") else "")
        out["ica_variance"] = ica.n_components
        out["ica_fit_highpass_hz"] = ica.info["highpass"]
    qc_path = dirs["qc"] / f"sub-{subject}_preprocessing_qc.json"
    if qc_path.exists():
        q = json.loads(qc_path.read_text())
        out["use_asr"] = bool(q.get("use_asr"))
        out["asr_cutoff"] = q.get("asr_cutoff")
    return out


def compare(subject: str, dirs: dict[str, Path]) -> list[dict[str, object]]:
    """Compare configured and applied settings for one subject.

    Parameters
    ----------
    subject : str
        Subject ID.
    dirs : dict of str to Path
        Dataset directories (uses ``"prep"`` and ``"qc"``).

    Returns
    -------
    list of dict
        Rows with keys subject, setting, configured, applied, match.
    """
    expected = expected_settings()
    applied = applied_settings(subject, dirs)
    rows = []
    for key, exp in expected.items():
        got = applied.get(key)
        if isinstance(exp, float) and got is not None:
            match = bool(np.isclose(float(got), exp))
        else:
            match = got == exp
        rows.append(
            {"subject": subject, "setting": key, "configured": exp, "applied": got, "match": match}
        )
    rows.append(
        {
            "subject": subject,
            "setting": "notch_db_at_line_freq",
            "configured": f"< {NOTCH_MAX_DB}",
            "applied": applied.get("notch_db"),
            "match": applied.get("notch_applied") is True,
        }
    )
    return rows


def assert_applied_matches(subject: str, dirs: dict[str, Path]) -> None:
    """Raise if any applied setting of one subject differs from the config.

    Parameters
    ----------
    subject : str
        Subject ID.
    dirs : dict of str to Path
        Dataset directories (uses ``"prep"`` and ``"qc"``).

    Raises
    ------
    AppliedSettingsMismatchError
        Listing every applied-vs-configured mismatch.
    """
    bad = [r for r in compare(subject, dirs) if not r["match"]]
    if bad:
        detail = "; ".join(
            f"{r['setting']}: configured {r['configured']}, applied {r['applied']}" for r in bad
        )
        raise AppliedSettingsMismatchError(
            f"sub-{subject}: applied settings differ from config -- {detail}"
        )
