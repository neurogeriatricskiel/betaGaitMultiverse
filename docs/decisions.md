# Analysis decisions

Current decisions and the reason for each, with the date it was made or last
changed. Superseded decisions and their history are in git; this file
describes only what the code and the stored results do now. Parameter values
are in `docs/parameters.md` (generated from `src/config.py`).

## Two datasets, two reference pipelines, one multiverse

- **Treadmill** (stepUpAms, key `stepup`): reference pipeline built from our
  own defensible choices.
- **Overground** (Jacobsen et al. 2020, OpenNeuro ds003039, key `jacobsen`):
  reference pipeline that reproduces the source study's preprocessing, with
  the four differences listed below (2026-08-11).
- The same 27-universe multiverse (high-pass x ASR x IC rule) runs on both.
  Robustness is shown by the effect holding across the same systematic
  multiverse in each dataset, not by forcing one pipeline onto both.

## Differences from the source study (overground reference)

1. **ICA:** Extended Infomax instead of AMICA (no maintained Python AMICA).
2. **Line noise:** 50 Hz notch filter instead of Zapline-plus (MATLAB only).
3. **ASR:** standard burst correction at SD 20 (asrpy) instead of the source
   study's channel-rejection-only call (FlatlineCriterion 5,
   ChannelCriterion 0.8, LineNoiseCriterion 4, burst correction off). There
   is no Python port of EEGLAB `clean_rawdata` channel rejection; decided
   2026-08-10 not to approximate it by hand.
4. **No spectral PCA** step (the source pipeline applies one before its GPM
   figures).

Matching the source study without change: 250 Hz resampling, 60 Hz low-pass,
average reference, ICA fitted on a 2 Hz high-passed copy of the data with the
weights applied to the 0.2 Hz data, ICLabel rule P(eye) > 0.9 or
P(muscle) > 0.9, fixed warp latencies [1, 18, 50, 68, 100] %.

## Treadmill reference

- **ASR off (as run, 2026-08-09).** Every treadmill preprocessing QC json has
  `use_asr: false`; t(16) = 5.13 is the ASR-off result. Since 2026-10-12,
  `REFERENCE_USE_ASR` per dataset is the default of `src.config.USE_ASR`;
  `BETAGAIT_USE_ASR` is only an explicit override.
- **High-pass 1.0 Hz, low-pass 60 Hz, balanced IC rule (P(brain) > 0.7).**
  These are the decision values of universe 11; the reference differs from
  universe 11 in cohort (n = 17 vs 15), TFR range (8-60 vs 8-40 Hz; no effect
  on the 13-30 Hz contrast) and float32 storage (below).
- **ICA fitted on the analysis data** (1.0 Hz; `ica.info["highpass"]` = 1.0 for
  all 17 saved ICAs, 2026-10-15). Only the overground reference fits ICA on a
  separate high-passed copy.
- **S21 excluded** (AutoReject leaves fewer than the 20 clean epochs the ICA
  fit needs under ASR off; numbers in `stepup_exclusions.tsv`).

## Overground reference

- **High-pass 0.2 Hz (rerun 2026-10-13/14).** The first run used 1.0 Hz
  because prepana02 passed a literal 1.0 Hz for both datasets. A pre-ICA
  comparison gave different signals for all 18 subjects and different bad
  channels for 9, so stages 2-7 were rerun at 0.2 Hz. prepana02 now applies
  each dataset's `REFERENCE_HIGHPASS_HZ` (`tests/test_prepana02_filter.py`)
  and stops if the saved raw, ICA or QC json differ from the configuration
  (`src/applied_settings.py`; `jacobsen_applied_settings.tsv`: 0 mismatches
  in 180 checks). Old vs new per subject and cohort:
  `jacobsen_highpass_1Hz_vs_0p2Hz.tsv` (primary t(13) 3.538 -> 3.501,
  +0.577 -> +0.582 dB).
- **The overground reference is not a multiverse universe.** Its 0.2 Hz
  high-pass is not a grid level (0.5 / 1 / 2 Hz), ICA is fitted on a 2 Hz
  copy, and cycles are warped to the fixed 18/50/68 % latencies. Its ASR (SD 20,
  burst correction, STAND calibration) is the multiverse "sd20" node and its
  eye/muscle rule is the multiverse "liberal" rule. Nearest universes:
  9 (HP 0.5 Hz, +0.577 dB) and 18 (HP 1.0 Hz, +0.551 dB); the reference
  gives +0.582 dB, above 24 of the 27 universes (range +0.22 to +0.68 dB).
- **Baseline:** `start_standing` -> `end_standing` (240 s for all 18; last 2 s
  trimmed, 238 s used), for both ASR calibration and ERSP normalisation; it
  replaced a fixed `start_restEEG` + 120 s window (2026-08-11).

## Duplicate recordings and cohorts (overground)

SHA-256 hashing and the OpenNeuro S3 ETags show three duplicate groups in
ds003039 as distributed: sub-005 = sub-007 and sub-009 = sub-013 (identical EEG
and events), and sub-006 / sub-008 (identical EEG, different events.tsv). All
18 subjects are processed; duplication is handled only when statistics are
reported:

- **Primary (n = 14):** drops one of each exact pair (007, 013) and both
  members of the inconsistent pair (006, 008). This is the headline result
  and the cohort of the overground figures and specification curve.
- **Full (n = 18)** and **minimal (n = 16;** drops 007 and 013 only) are
  sensitivity variants.

Source of truth: `results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt`
(written by `scripts/qc/qc_jacobsen_duplicate_audit.py`); cohort flow in
`docs/cohorts.md`.

## Gait events and time warping

- Cycles are RHS to RHS, warped on all four events (RHS, LTO, LHS, RTO) so a
  given percentage means the same gait phase in every stride. Double stance:
  RHS -> LTO and LHS -> RTO; swing: LTO -> LHS and RTO -> RHS.
- **Frozen group-median anchors** (`group_gait_event_anchors_frozen.json`,
  frozen 2026-08-07; written only by `scripts/freeze_gait_anchors.py`, run by
  hand): treadmill 10.00 / 50.00 / 59.81 %, overground 15.98 / 50.28 / 65.98 %.
  Used by the treadmill reference and both multiverses. The treadmill values
  were pooled from the cycle tables of the earlier ASR SD 20 run (17 subjects
  including S21, without S2; 4,822 cycles); the current tables (with S2,
  without S21; 4,792 cycles) give 10.09 / 50.00 / 60.00 % (2026-10-15). The
  frozen values are kept so the stored ERSP and multiverse results stay
  comparable. Sensitivity (scratch rerun of stages 5-7 with the recomputed
  anchors, 2026-10-16): t(16) = 5.098, p = 1.07e-4, mean difference
  +1.928 dB (stored 5.129, +1.949 dB); largest per-subject change S10,
  -0.058 dB; S2 moves from +0.001 to -0.010 dB. Anchor timings
  come from gait events, which do not depend on EEG preprocessing; ASR only
  changes which cycles survive.
- **Overground reference** uses the source study's fixed 18 / 50 / 68 %.

## ERSP, baseline and averaging order

- **Standing baseline only** for both datasets (2026-08-11): dB re mean power
  of the standing segment (2 s epochs, linear mean, last 2 s trimmed). The
  double-stance - swing contrast is invariant to any per-subject,
  per-frequency constant, so gait power modulation (GPM, ERSP minus its
  cycle mean) gives the same contrast; GPM appears only as a supplementary
  figure for comparison with the source study.
- **Averaging order: mean over cycles of 10*log10(P / P_standing)** (log then
  average; Makeig 1993, Delorme & Makeig 2004). Kept (2026-09-16). The source
  study averages magnitude first and then takes the log. On the 0.2 Hz data
  (`jacobsen_ersp_averaging_order.txt`, 2026-10-15):
  - the group beta map is entirely below baseline in our order (mean
    -2.10 dB, primary) and almost entirely above it in the source order
    (+0.76 dB); 13 of 14 primary subjects change sign;
  - the double-stance - swing contrast barely changes: primary t(13) = 3.50
    (ours) vs 3.44 (source), full t(17) = 3.97 vs 3.91.
  - Stationary-noise simulation (`jacobsen_log_averaging_bias.txt`): with no
    true change, our order gives -2.52 to -2.56 dB, the expected log-averaging
    bias 10*log10(e) x (-0.5772) = -2.51 dB; the source order gives about 0 dB.
  - Consequence for the manuscript: the double-stance - swing contrast is the
    robust finding; the absolute ERD / ERS level of overground walking depends
    on the estimator and is reported with that caveat.
  - Also checked (2026-09-16, on the earlier 1.0 Hz data) and ruled out as
    causes of the all-negative overground map: beta band 13-30 vs 20-30 Hz,
    and power vs magnitude baseline convention.
- **Beta sub-bands** (`jacobsen_gpm_band_check.txt`): primary 20-30 Hz
  t(13) = 3.57 and 13-20 Hz t(13) = 3.26 vs 13-30 Hz t(13) = 3.50.

## Channels and ROI

- **Treadmill:** 64 EEG channels analysed. The BIDS channels.tsv also types
  the STATUS and "Counter 2power24" trigger channels as EEG; they are dropped
  on loading. M1 / M2 have standard_1005 positions and enter the average
  reference and ICA; M1 is the farthest channel from Cz (19.25 cm, ROI
  weight 0), M2 has weight 0.00008.
- **Overground:** 64 EEG channels (+3 accelerometer channels, dropped);
  positions from each recording's electrodes.tsv. FT9 / TP9 / FT10 / TP10 are
  the farthest channels from Cz (15.85 cm) and get weight 0.
- **ROI:** linear distance weights centred on Cz, w_i = 1 - d_i / d_max,
  normalised to sum 1.

## Multiverse

- **Grid (Table 2.1):** high-pass 0.5 / 1 / 2 Hz; ASR off / SD 3 / SD 20;
  IC rule conservative (keep P(brain) > 0.9) / balanced (> 0.7) / liberal
  (reject P(muscle) or P(eye) > 0.9). Fixed: low-pass 60 Hz, TFR 8-40 Hz,
  average reference, Extended Infomax, frozen group anchors.
- **ICA cached per (high-pass, ASR) branch** (9 branches); the three IC rules
  reuse it. The ICA is fitted on that universe's data (no separate high-passed
  copy) in both datasets.
- **Cohorts:** treadmill n = 15 (`MULTIVERSE_SUBJECTS`; excludes S21, S10
  with 1 of 38 brain ICs, and S2, whose AutoReject fails in some branches);
  overground: all 18 computed, specification curve on the primary n = 14
  (`dedup_diffs`).
- **Statistics per universe:** one-sample t-test of the per-subject contrasts
  against 0 (two-sided, uncorrected for the 27 universes), t-based 95 % CI.
  27 / 27 universes p < 0.05 in both datasets.
- **Variance decomposition** (eta squared of the per-universe t, 2026-09-16):
  treadmill ASR 0.53, IC rule 0.13, high-pass 0.03 (ASR dominant); overground
  IC rule 0.38, ASR 0.34, high-pass 0.10 (no single dominant node). Only the
  within-dataset ranking is comparable, not the values across datasets.
- **Low-pass 40 Hz (archived run):** the treadmill multiverse was first run
  with a 40 Hz low-pass. It is kept read-only and reported as a sensitivity
  table and specification curve (`results/supplementary/stepup_*lowpass40*`);
  n = 15 in 26 universes and 14 in one.
- **GEDAI not included:** evaluated as a candidate node on three treadmill
  participants (S1, S9, S14) and not included. The first gate figure (96.3 %
  of Cz-ROI 13-30 Hz power removed, mean Welch over the whole recording) is
  superseded: a filter transient at the STAND/CS join inflated the
  whole-recording "before" power (14x in S1, 49x in S9;
  `scripts/qc/qc_join_transient.py`). Corrected values are in
  `results/pipeline/stepup/qc/stepup_gedai_gate.tsv` (segments without
  20 s at the join, median Welch, GEDAI fitted on the whole recording and
  on each segment alone): [TO FILL after S14]. The 15 % limit is our own
  criterion, not taken from the literature. A reduction in beta power does
  not by itself separate artifact from neural activity. Kept only as a
  documented methods check (`src/nodes/gedai_node.py`,
  `scripts/qc/qc_gedai_gate.py`, `stepup_gedai_gate_check.txt`).

## Storage, determinism and reproducibility

- AutoReject and ICA use `RANDOM_STATE = 42`. Rerunning stages 1-5 for S1 and
  stages 3-5 for sub-001 reproduces the stored files bit for bit (2026-10-15).
- The reference pipeline saves intermediate `.fif` files as float32 (MNE
  default) and reloads them stage by stage; the multiverse stays in float64 in
  memory. In a pre-flight with matching settings this gave a ~0.9 % difference
  in t (2026-08-08); accepted as immaterial, not fixed.
- `run_ica` requires `fit_params` (Extended Infomax is passed everywhere).

## Known limitations

- Overground: turns within the outdoor walking bout cannot be detected or
  excluded (no heading or turn markers in ds003039).
- Bad channels interpolated per subject are printed by prepana02 but not
  stored.
- The scripts behind the treadmill S21 AutoReject recheck and the S2 branch
  check were not preserved; their outputs are in `datasets/stepup/d00_qc/` and
  quoted in `stepup_exclusions.tsv`.
- The stored universe scripts predate the `subjects_ok` field (treadmill) and
  `dedup_diffs` was added to the overground results afterwards; regenerating
  the universe scripts does not reproduce the stored pkl format exactly.
