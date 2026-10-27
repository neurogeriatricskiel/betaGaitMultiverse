# Treadmill (stepup) QC and statistics

Scripts are in `scripts/` (stage scripts) and `scripts/qc/`. Manuscript
sections: M = Methods, R = Results, S = Supplement.

| File | Contents | Written by | Used in |
|---|---|---|---|
| `stepup_betaphase_stats.txt` | Reference statistic: per-subject double-stance and swing beta (ROI, dB); t(16) = 5.13, n = 17 | `prepana07_betaphase_stats.py` | R, reference statistic |
| `stepup_subject_qc.tsv` | Per subject: channels, applied high-pass, ASR, ICA and brain components, AutoReject epochs, gait cycles detected / extracted / used, worst QC flag | `qc/qc_build_tables.py` | S, QC table |
| `stepup_exclusions.tsv` | Excluded subjects with stage, reason, numbers and source: S21 (AutoReject), S10 (ICA), S2 (multiverse branches) | `qc/qc_build_tables.py` | M2.1, cohorts |
| `stepup_cohorts.json` | Configured (18), reference (17) and multiverse (15) cohorts | `qc/qc_build_tables.py` | M2.1, cohorts |
| `stepup_parameters_used.json` | Every resolved `src.config` value for this dataset (paths relative to the repository root) | `qc/qc_build_tables.py` | M2, parameters |
| `stepup_multiverse_universes.tsv` | Per universe: decisions, n, mean dB, t, p, 95 % CI (the specification-curve numbers) | `qc/qc_build_tables.py` | R, specification curve |
| `stepup_qc_summary.tsv`, `stepup_qc_flags.tsv` | Every per-subject QC json; QC flag per subject and stage | `qc/qc_build_tables.py` | S, QC table |
| `stepup_applied_settings.tsv` | Per subject and setting: configured vs applied high-pass, low-pass, sampling rate, 50 Hz notch, ASR, ICA method / variance / fit-data high-pass (`ica.info["highpass"]`), match flag | `qc/qc_applied_settings.py` | M2.3-2.4 |
| `stepup_baseline_window.tsv` | Per subject: STAND and CS recording length, baseline used (STAND minus 2 s) | `qc/qc_baseline_window.py` | M2.5, baseline |
| `stepup_ersp_averaging_order.txt` | Beta ERSP (ROI) under our vs the source study's averaging order, reference cohort n = 17: group-map range, mean, % negative, maximum; subjects whose mean beta ERSP changes sign (10 of 17); double stance vs swing t per order (5.13 vs 5.21) | `qc/qc_averaging_order.py stepup` (~25 min) | D, absolute ERD level |
| `../../../multiverse/stepup/outputs/stepup_universe_subject_contrasts.tsv` | One row per universe and subject (15 subjects): decisions, beta difference (dB), cohort flags; double stance and swing are empty because the treadmill results store only the differences. Recomputing n, mean, t, p and CI from it reproduces the specification curve | `qc/qc_universe_subject_contrasts.py` | R, specification curve (data) |
| `stepup_band_diagnostics.txt` | Local diagnostic, not tracked | `scripts/diag/` (not tracked) | - |
| `stepup_gedai_gate_check.txt`, `stepup_gedai_gate.tsv` | GEDAI gate. Primary (gating) measure: % Cz-ROI 13-30 Hz power removed in the walking segment without its first 20 s, Welch average median, GEDAI fitted on the whole recording, broadband multiplier 3, mean over S1, S9, S14; limit 15 % (our criterion). Sensitivity (not gating): separate fit, multiplier 6 (node default), mean Welch, standing without its last 20 s. Superseded: the original whole / walking / standing rows (mean Welch; whole recording inflated by the join transient). Column `role`; absolute ROI beta before and after; variance removed; gedai version and `gedai_node.py` hash | `qc/qc_gedai_gate.py` | M2.7 (GEDAI exclusion) |
| `stepup_join_transient.txt`, `stepup_join_transient.tsv` | STAND/CS join (files concatenated before the 1 Hz high-pass): per-second Cz-ROI variance ratio near the join, cycles near walk onset, contrast without cycles in the first 20 s of walking (group t), bad channels with the join masked, join epoch in the ICA-fit epochs; multiverse 0.5 Hz universes for two subjects | `qc/qc_join_transient.py` | M2.3 (preprocessing) |
| `stepup_roi_channel_sensitivity.txt` | ROI weights of Fp*/AF* (7.0 % of the weight); per-channel double stance - swing (largest Fz +4.78 dB) and contribution to the ROI contrast (Fp*/AF* 13.3 %); ROI without Fp*/AF* t(16) = 5.02, without frontal and lateral 5.05, narrow 5-channel ROI 4.42 (+1.15 dB); frequency profile at Fpz and Cz | `qc/qc_roi_channel_sensitivity.py` | S, ROI sensitivity; D, frontal channels |
| `stepup_channel_contrast.tsv`, `stepup_freq_profile_fpz_cz.png` | Per channel: weight, distance to Cz, mean contrast, t, p, contribution; ERSP by frequency at Fpz and Cz | `qc/qc_roi_channel_sensitivity.py` | S, D |
| `stepup_ica_audit.txt` | ICLabel recomputed from the saved ICA (exclusions equal the saved ones for all 17); removed components by class; retained components with P(eye) or P(muscle) > 0.3 (none); Fpz 13-30 Hz walking variance kept after ICA (median 66 %); reference without S10 t(15) = 4.74 | `qc/qc_ica_audit.py` | D, frontal channels and ICA |
| `stepup_heelstrike_locked.txt`, `.png` | Heel-strike-locked 1-40 Hz average at Fpz and Cz, pre-ICA and ICA-corrected (Fpz 26.4 uV vs Cz 5.2 uV peak-to-peak after ICA) | `qc/qc_heelstrike_locked.py` | D, movement artefact |
| `stepup_multiverse_frontal.txt`; `results/multiverse/stepup/outputs/stepup_multiverse_frontal.tsv` | Per universe (n = 15, from the float32 ICA-corrected caches): heel-strike-locked Fpz and Cz peak-to-peak (1-40 Hz), ROI contrast with the stored ROI, without Fp*/AF* and for the five channels nearest Cz (dB, t, p), next to the stored contrast; correlation of each deflection with the stored contrast across universes and within each ASR mode (universes share participants: descriptive only) | `qc/qc_multiverse_frontal.py stepup` | D, movement artefact across the multiverse |

Notes

- The S21 rows in the subject and summary tables come from QC jsons of an
  earlier run with ASR SD 20; the current reference run (ASR off) fails at
  AutoReject for S21 (`stepup_exclusions.tsv`).
- Bad channels interpolated per subject are printed by prepana02 but not
  stored ("not recorded" in the subject table).
- The scripts behind the S21 AutoReject recheck and the S2 branch check were
  not preserved; their outputs are `datasets/stepup/d00_qc/sub-S21_autoreject_recheck_qc.json`
  and `sub-S2_multiverse_branch_autoreject_check_qc.json`, quoted in `stepup_exclusions.tsv`.
- Decisions and their reasons: `docs/decisions.md`; cohorts: `docs/cohorts.md`.
