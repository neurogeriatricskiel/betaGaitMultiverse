"""
qc_log_averaging_bias.py -- stationary-noise ERSP under both averaging orders.

Purpose
    Stationary Gaussian noise (white and 1/f), so walking and standing have
    the same true power and the true ERSP is 0 dB. A standing part
    (STAND annotation, BASELINE_EPOCH_DUR_S epochs, linear mean power, as in
    the pipeline) and a contiguous walking part cut into gait cycles whose
    lengths and LTO/LHS/RTO times are drawn from the overground cycle
    tables. Each cycle goes through the pipeline's per-cycle loop
    (src.averaging_order: Morlet, n_cycles = f / 2, 8-60 Hz, 5 % edge crop,
    warp to the fixed overground anchors) and gives beta (13-30 Hz) ERSP in
      ours   : mean_k 10*log10(P_k / P_standing)
      source : 20*log10(mean_k |X_k| / mean |X_standing|)
    For power that is exponentially distributed (|X| Rayleigh), the mean of
    10*log10(P / E[P]) is 10*log10(e) * (-0.5772) = -2.51 dB (Euler-Mascheroni
    constant), while the ratio of mean magnitudes is unbiased. Reports the
    mean ERSP per order and noise type over 20 seeds.
Inputs
    datasets/jacobsen/d04_gaitepochs/sub-*_cycles_kept.tsv (cycle lengths and event times only)
Outputs
    results/pipeline/jacobsen/qc/jacobsen_log_averaging_bias.txt
Manuscript
    Discussion / supplement (why the overground ERSP level depends on averaging order).
Run
    python scripts/qc/qc_log_averaging_bias.py   (always overground; ~10 min)
"""

import os

os.environ["BETAGAIT_DATASET"] = "jacobsen"

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.averaging_order import (  # noqa: E402
    FREQS,
    cycle_ersp_orders,
    standing_power_and_magnitude,
)
from src.config import (  # noqa: E402
    BASELINE_EPOCH_DUR_S,
    DIR_QC,
    SUBJECTS,
    TARGET_SFREQ,
    TFR_EDGE_CROP,
)
from src.ersp import BETA_FMAX, BETA_FMIN, load_reference_anchors  # noqa: E402
from src.paths import get_dataset_dirs  # noqa: E402

SEEDS = range(20)
N_CHANNELS = 8
STAND_S = 240.0  # s, the overground start_standing -> end_standing window
N_CYCLES = 230  # cycles per simulated subject (overground: 103-281, mean 215)
NOISE_SD = 10e-6  # V; far below AMP_THRESH, so no epoch or cycle is rejected
EULER_GAMMA = 0.5772156649015329
EXPECTED_DB = 10 * np.log10(np.e) * -EULER_GAMMA
BETA = (FREQS >= BETA_FMIN) & (FREQS <= BETA_FMAX)


def cycle_table(dirs):
    """Pooled overground cycles: duration and event times relative to RHS (s)."""
    rows = []
    for s in SUBJECTS:
        m = pd.read_csv(dirs["gaitepochs"] / f"sub-{s}_cycles_kept.tsv", sep="\t")
        rows.append(
            pd.DataFrame(
                {
                    "dur": m["rhs_end_s"] - m["rhs_start_s"],
                    **{k: m[f"{k}_s"] - m["rhs_start_s"] for k in ("lto", "lhs", "rto")},
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def noise(rng, kind, n_samples):
    """(N_CHANNELS, n_samples) stationary Gaussian noise, white or 1/f power, SD NOISE_SD."""
    x = rng.standard_normal((N_CHANNELS, n_samples))
    if kind == "1/f":
        spec = np.fft.rfft(x, axis=-1)
        f = np.fft.rfftfreq(n_samples, d=1 / TARGET_SFREQ)
        shape = np.zeros_like(f)
        shape[1:] = 1 / np.sqrt(f[1:])  # amplitude 1/sqrt(f) -> power 1/f
        x = np.fft.irfft(spec * shape, n=n_samples, axis=-1)
    return x / x.std(axis=-1, keepdims=True) * NOISE_SD


def simulate(seed, kind, cycles, anchors):
    """Channel-mean beta ERSP (dB) of one simulated subject, (ours, source)."""
    rng = np.random.default_rng(seed)
    sf = float(TARGET_SFREQ)
    drawn = cycles.iloc[rng.integers(0, len(cycles), N_CYCLES)].reset_index(drop=True)
    starts = np.concatenate([[0.0], np.cumsum(drawn["dur"].to_numpy())[:-1]])
    meta = pd.DataFrame(
        {
            "rhs_start_s": starts,
            "rhs_end_s": starts + drawn["dur"],
            **{f"{k}_s": starts + drawn[k] for k in ("lto", "lhs", "rto")},
        }
    )
    walk_s = float(meta["rhs_end_s"].iloc[-1]) + 1.0
    x = noise(rng, kind, int(round((STAND_S + walk_s) * sf)))
    n_stand = int(round(STAND_S * sf))

    info = mne.create_info([f"EEG{i:02d}" for i in range(N_CHANNELS)], sf, "eeg")
    stand = mne.io.RawArray(x[:, :n_stand], info, verbose=False)
    stand.set_annotations(mne.Annotations(0.0, STAND_S, "STAND"))
    bp, bm, _ = standing_power_and_magnitude(stand)

    # Cycles cut as in prepana04 (RHS to RHS, sample index = round(t * sfreq)).
    walk = x[:, n_stand:]
    segs = [
        walk[:, int(round(r.rhs_start_s * sf)) : int(round(r.rhs_end_s * sf))]
        for r in meta.itertuples()
    ]
    ours, source, n = cycle_ersp_orders(segs, list(range(N_CHANNELS)), sf, meta, bp, bm, anchors)
    if n != N_CYCLES:
        raise SystemExit(f"seed {seed} {kind}: {N_CYCLES - n} cycles dropped")
    return float(ours[:, BETA].mean()), float(source[:, BETA].mean())


def main() -> None:
    """Simulate every seed and noise type and write the bias report."""
    mne.set_log_level("ERROR")
    dirs = get_dataset_dirs("jacobsen")
    anchors = load_reference_anchors(dirs["ersp"])
    cycles = cycle_table(dirs)
    lines = [
        "ERSP of stationary Gaussian noise (true change 0 dB): beta 13-30 Hz, channel mean",
        f"  {len(SEEDS)} seeds x {N_CHANNELS} channels; standing {STAND_S:g} s "
        f"({BASELINE_EPOCH_DUR_S:g} s epochs, linear mean power); {N_CYCLES} cycles per seed "
        f"drawn from {len(cycles)} overground cycles (duration {cycles['dur'].min():.2f}-"
        f"{cycles['dur'].max():.2f} s, median {cycles['dur'].median():.2f} s); "
        f"Morlet n_cycles = f/2, {FREQS[0]:g}-{FREQS[-1]:g} Hz, {TFR_EDGE_CROP:.0%} edge crop, "
        f"{TARGET_SFREQ} Hz; warp anchors {tuple(float(a) for a in anchors)}",
        f"  Expected bias of the mean of log power: 10*log10(e)*(-0.5772) = {EXPECTED_DB:.3f} dB",
        "",
    ]
    for kind in ("white", "1/f"):
        res = np.array([simulate(seed, kind, cycles, anchors) for seed in SEEDS])
        ours, source, gap = res[:, 0], res[:, 1], res[:, 0] - res[:, 1]
        lines.append(f"{kind} noise:")
        for name, v in (("ours", ours), ("source", source), ("ours - source", gap)):
            lines.append(
                f"  {name:<13s}: mean {v.mean():+.3f} dB  SD {v.std(ddof=1):.3f}  "
                f"range {v.min():+.3f} to {v.max():+.3f} dB  (vs {EXPECTED_DB:.3f}: "
                f"{v.mean() - EXPECTED_DB:+.3f})"
            )
        lines.append("")
    text = "\n".join(lines).rstrip()
    print(text)
    out = DIR_QC / "jacobsen_log_averaging_bias.txt"
    out.write_text(text + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
