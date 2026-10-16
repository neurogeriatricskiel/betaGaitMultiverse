# Overground (jacobsen) QC and statistics

Scripts are in `scripts/` (stage scripts) and `scripts/qc/`. Manuscript
sections: M = Methods, R = Results, D = Discussion, S = Supplement.

| File | Contents | Written by | Used in |
|---|---|---|---|
| `jacobsen_duplicate_recordings.txt` | Duplicate recordings and the three cohorts (FULL n = 18, PRIMARY n = 14, MINIMAL n = 16) with their statistics; PRIMARY t(13) = 3.50 is the headline overground result. Every script reads the cohorts from this file | `qc/qc_jacobsen_duplicate_audit.py` | M2.1, R |
| `jacobsen_dataquality_audit.txt` | SHA-256 hashes, S3 ETag check, load integrity and EEGLAB metadata of all 19 downloaded subjects (Checks 1-4) | `qc/qc_jacobsen_duplicate_audit.py` | M2.1 |
| `jacobsen_betaphase_stats.txt` | Reference statistic on all 18 subjects, t(17) = 3.97 (FULL cohort) | `prepana07_betaphase_stats.py` | R, sensitivity |
| `jacobsen_subject_qc.tsv` | Per subject: channels, applied high-pass, ASR, ICA and brain components, AutoReject epochs, gait cycles detected / extracted / used, worst QC flag | `qc/qc_build_tables.py` | S, QC table |
| `jacobsen_exclusions.tsv` | sub-019 (truncated at source) and the duplicates dropped from the primary cohort | `qc/qc_build_tables.py` | M2.1, cohorts |
| `jacobsen_cohorts.json` | Configured, reference, multiverse, primary and minimal cohorts | `qc/qc_build_tables.py` | M2.1, cohorts |
| `jacobsen_parameters_used.json` | Every resolved `src.config` value for this dataset (paths relative to the repository root) | `qc/qc_build_tables.py` | M2, parameters |
| `jacobsen_multiverse_universes.tsv` | Per universe on the primary cohort: decisions, n, mean dB, t, p, 95 % CI (the specification-curve numbers) | `qc/qc_build_tables.py` | R, specification curve |
| `jacobsen_qc_summary.tsv`, `jacobsen_qc_flags.tsv` | Every per-subject QC json; QC flag per subject and stage | `qc/qc_build_tables.py` | S, QC table |
| `jacobsen_applied_settings.tsv` | Per subject and setting: configured vs applied high-pass, low-pass, sampling rate, 50 Hz notch, ASR, ICA method / variance / fit-data high-pass (`ica.info["highpass"]`), match flag; 0 mismatches | `qc/qc_applied_settings.py` | M2.3-2.4 |
| `jacobsen_baseline_window.tsv` | Per subject: start_standing / end_standing present once, 240 s window, ends before walking; baseline used 238 s | `qc/qc_baseline_window.py` | M2.5, baseline |
| `jacobsen_highpass_1Hz_vs_0p2Hz.tsv` | First run (1.0 Hz high-pass) vs rerun (0.2 Hz): per subject bad channels, ICA / brain components, AutoReject epochs, cycles, beta double stance, swing, contrast; per cohort n, df, t, p, mean difference, n with DS > swing | one-off scratch script, 2026-10-15 (the 1.0 Hz outputs are not in the repo; bad channels from the run logs) | M2.3, rerun note |
| `jacobsen_gpm_averaging_order.txt` (+ `.npz`, `.png`) | Group GPM under our averaging order vs the source study's (pixel corr 0.998, amplitude ratio 0.97) and the double stance vs swing t per order, FULL and PRIMARY | `qc/qc_averaging_order.py` (~25 min) | D, comparison with the source study |
| `jacobsen_ersp_averaging_order.txt` | Beta ERSP (ROI) under both averaging orders: group-map range, mean, % negative, maximum; subjects whose mean beta ERSP changes sign (13 of 14 primary); double stance vs swing t per order (primary 3.50 vs 3.44) | `qc/qc_averaging_order.py` | D, absolute ERD level |
| `jacobsen_log_averaging_bias.txt` | Stationary-noise simulation (white and 1/f, 20 seeds) through the pipeline's baseline and per-cycle steps: our order -2.52 to -2.56 dB vs the expected -2.51 dB log-averaging bias; source order about 0 dB | `qc/qc_log_averaging_bias.py` (~10 min) | D, absolute ERD level |
| `jacobsen_gpm_band_check.txt` | GPM peak, trough and double stance - swing for 13-20, 20-30, 10-17, 20-35 Hz (ROI and Cz); paired t for 13-30, 13-20, 20-30 Hz (primary 20-30 Hz t(13) = 3.57) | `qc/qc_gpm_band_check.py` | D, beta sub-bands |
| `../../../multiverse/jacobsen/outputs/jacobsen_universe_subject_contrasts.tsv` | One row per universe and subject (18 subjects): decisions, beta double stance, swing and difference (dB), cohort flags (`in_speccurve_cohort` = primary). Recomputing n, mean, t, p and CI from it reproduces the specification curve | `qc/qc_universe_subject_contrasts.py` | R, specification curve (data) |
| `jacobsen_baseline_120s_check.txt` | Check of a retired 120 s restEEG baseline (not tracked) | `scripts/diag/` | - |
| `jacobsen_roi_channel_sensitivity.txt` | ROI weights of Fp*/AF* (7.2 % of the weight); per-channel double stance - swing and contribution to the ROI contrast (Fp*/AF* 8.6 %); ROI without Fp*/AF* t(13) = 3.51, without frontal and lateral 3.49, narrow 5-channel ROI 3.44; frequency profile at Fpz and Cz | `qc/qc_roi_channel_sensitivity.py` | S, ROI sensitivity |
| `jacobsen_channel_contrast.tsv`, `jacobsen_freq_profile_fpz_cz.png` | Per channel: weight, distance to Cz, mean contrast, t, p, contribution; ERSP by frequency at Fpz and Cz | `qc/qc_roi_channel_sensitivity.py` | S |
| `jacobsen_ica_audit.txt` | As the treadmill file, primary cohort n = 14 (exclusions equal the saved ones for all 14), rule P(eye) or P(muscle) > 0.9, so retained components with P(muscle) > 0.3 are expected; Fpz 13-30 Hz walking variance kept after ICA median 38 % | `qc/qc_ica_audit.py` (~1 min per subject) | D, frontal channels and ICA |
| `jacobsen_heelstrike_locked.txt`, `.png` | Heel-strike-locked 1-40 Hz average at Fpz and Cz, pre-ICA and ICA-corrected (Fpz 5.6 uV vs Cz 11.8 uV peak-to-peak after ICA) | `qc/qc_heelstrike_locked.py` | D, movement artefact |
| `jacobsen_multiverse_frontal.txt`; `results/multiverse/jacobsen/outputs/jacobsen_multiverse_frontal.tsv` | Per universe (primary n = 14, from the float32 ICA-corrected caches): heel-strike-locked Fpz and Cz peak-to-peak (1-40 Hz), ROI contrast with the stored ROI, without Fp*/AF* and for the five channels nearest Cz (dB, t, p), next to the stored contrast; correlation of each deflection with the stored contrast across universes and within each ASR mode (universes share participants: descriptive only) | `qc/qc_multiverse_frontal.py jacobsen` | D, movement artefact across the multiverse |

Notes

- The overground reference was rerun at 0.2 Hz on 2026-10-13/14 (the first
  run used 1.0 Hz; see `docs/decisions.md`).
- Bad channels interpolated per subject are printed by prepana02 but not
  stored ("not recorded" in the subject table).
- Decisions and their reasons: `docs/decisions.md`; cohorts: `docs/cohorts.md`.
