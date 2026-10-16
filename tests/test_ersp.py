"""Warp, phase masks and the double-stance - swing contrast on synthetic data."""

import numpy as np
import pytest

from src.ersp import phase_split_indices, warp_cycle_to_grid

ANCHORS = (16.0, 50.0, 66.0)


def test_phase_masks_partition_grid():
    ds, sw = phase_split_indices(ANCHORS, n_points=101)
    assert np.array_equal(np.sort(np.concatenate([ds, sw])), np.arange(101))
    assert len(np.intersect1d(ds, sw)) == 0
    assert ds[0] == 0 and 16 not in ds and 16 in sw  # LTO starts swing
    assert 50 in ds and 66 in sw  # LHS starts DS, RTO starts swing


def test_warp_pins_events_to_anchors():
    n_t, events = 240, (30, 115, 150)
    # Time axis encoded as the signal: value = sample index.
    power = np.tile(np.arange(n_t, dtype=float), (2, 3, 1))
    out = warp_cycle_to_grid(power, *events, anchors=ANCHORS, n_points=101)
    assert out.shape == (2, 3, 101)
    for event, anchor in zip(events, ANCHORS, strict=True):
        assert out[0, 0, int(anchor)] == pytest.approx(event)
    assert out[0, 0, 0] == 0 and out[0, 0, 100] == n_t - 1
    assert np.all(np.diff(out[0, 0]) > 0)  # monotonic


def test_warp_rejects_bad_event_order():
    power = np.zeros((1, 1, 100))
    with pytest.raises(ValueError):
        warp_cycle_to_grid(power, 50, 40, 60, anchors=ANCHORS)
    with pytest.raises(ValueError):
        warp_cycle_to_grid(power, 10, 40, 99, anchors=ANCHORS)


def test_contrast_invariant_to_per_frequency_constant():
    rng = np.random.default_rng(0)
    ersp = rng.normal(size=(18, 101))  # (n_freqs, grid)
    ds, sw = phase_split_indices(ANCHORS, n_points=101)

    def contrast(x):
        return x[:, ds].mean() - x[:, sw].mean()

    shifted = ersp + rng.normal(size=(18, 1)) * 5  # e.g. a different baseline per frequency
    gpm = ersp - ersp.mean(axis=-1, keepdims=True)  # GPM
    assert contrast(shifted) == pytest.approx(contrast(ersp))
    assert contrast(gpm) == pytest.approx(contrast(ersp))
