# Cohorts

Short cohort flow per dataset. The source of truth is, per dataset,
`results/pipeline/<dataset>/qc/<dataset>_cohorts.json` (subject lists and n)
and `<dataset>_exclusions.tsv` (reason, numbers and source of every
exclusion); both are written by `scripts/qc/qc_build_tables.py` from the
stored data. Per-subject counts are in `<dataset>_subject_qc.tsv`.

## Treadmill (stepup)

| Stage | n | Change |
|---|---|---|
| Raw dataset (`datasets/stepup/d00_raw/`) | 18 | - |
| Configured (`SUBJECTS`) | 18 | - |
| Reference pipeline and statistic (prepana05-07) | 17 | S21 excluded: AutoReject leaves fewer than 20 clean epochs under ASR off |
| Multiverse (`MULTIVERSE_SUBJECTS`) | 15 | also S10 (ICA: 1 of 38 brain ICs) and S2 (AutoReject fails in some branches) |

Gait cycles used (reference, n = 17): 4792 in total, 281.9 per subject
(264-302).

## Overground (jacobsen, OpenNeuro ds003039)

| Stage | n | Change |
|---|---|---|
| Downloaded (`datasets/jacobsen/d00_raw/`) | 19 | - |
| Configured (`SUBJECTS`) | 18 | sub-019 excluded: .fdt truncated at the source |
| Reference pipeline, full statistic (prepana05-07) | 18 | duplicates kept |
| Primary cohort (headline statistic, figures, specification curve) | 14 | drops 007 (= 005), 013 (= 009) and both of 006 / 008 (same EEG, different events) |
| Minimal sensitivity cohort | 16 | drops 007 and 013 only |
| Multiverse | 18 computed | reported on the primary cohort (n = 14, `dedup_diffs`) |

Cohort rule and evidence: `results/pipeline/jacobsen/qc/jacobsen_duplicate_recordings.txt`
and `jacobsen_dataquality_audit.txt` (`scripts/qc/qc_jacobsen_duplicate_audit.py`).

Gait cycles used: full cohort 3865 in total, 214.7 per subject (103-281);
primary cohort 3197 in total, 228.4 per subject (202-281).

Demographics: the overground `participants.tsv` gives sex, age (19-32 years)
and handedness (all right-handed); the treadmill `participants.tsv` holds only
the participant IDs. The exact duplicate sub-005 / sub-007 is listed with
different ages (21 and 32) despite identical EEG and events, which points to
a labelling error at the source.
