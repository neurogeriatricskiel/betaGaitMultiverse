"""
Preprocessing utilities for EEG data.

Covers every step from raw segmented data up to pre-ICA clean epochs.
"""

import mne
import numpy as np


def drop_invalid_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Drop non-EEG bookkeeping channels (status, counter, source) by name.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording; modified in place.

    Returns
    -------
    mne.io.BaseRaw
        The same object, with matching channels dropped.
    """
    invalid_labels = ["status", "counter", "counter 2power24", "source"]

    to_drop = [ch for ch in raw.ch_names if any(bad in ch.lower() for bad in invalid_labels)]

    if len(to_drop) > 0:
        print(f"Dropping invalid channels: {to_drop}")
        raw.drop_channels(to_drop)
    else:
        print("No invalid channels found.")

    return raw


def drop_invalid_eeg_channels(epochs: mne.Epochs) -> mne.Epochs:
    """Drop EEG channels with NaN or all-zero location vectors.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to check.

    Returns
    -------
    mne.Epochs
        Epochs without the invalid channels (unchanged if none).
    """
    bad_channels = []

    for ch in epochs.info["chs"]:
        if ch["kind"] == 2:  # EEG
            loc = ch["loc"][:3]

            if np.any(np.isnan(loc)) or np.all(loc == 0.0):
                bad_channels.append(ch["ch_name"])

    if bad_channels:
        print(f"  Dropping {len(bad_channels)} invalid EEG channels: {bad_channels}")
        return epochs.drop_channels(bad_channels)
    return epochs
