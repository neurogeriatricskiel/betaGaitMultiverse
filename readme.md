# Multiverse analysis of beta activity during gait

Does sensorimotor beta power (13-30 Hz) differ between double stance and
swing during walking, and does the answer survive defensible changes to EEG
preprocessing? This repository computes the per-subject double-stance - swing
beta contrast in two independent datasets (treadmill and overground walking)
with a fixed reference pipeline per dataset, and repeats it over the same
27-universe preprocessing multiverse (high-pass x ASR x IC rule) in both
(Steegen et al. 2016; Clayson et al. 2021).

Main results: treadmill t(16) = 5.13 (n = 17); overground primary cohort
t(13) = 3.50 (n = 14 independent recordings); 27 / 27 universes p < 0.05 in
each dataset (uncorrected).

## Datasets

| Key | Label | Source | Access | Format, events |
|---|---|---|---|---|
| `stepup` | treadmill | stepUpAms, VU Amsterdam: treadmill walking, separate standing (STAND) and constant-speed walking (CS) recordings | not distributed with this repository (see Open items) | BIDS BrainVision; gait events detected from optical motion capture |
| `jacobsen` | overground | Jacobsen et al. 2020, OpenNeuro [ds003039](https://openneuro.org/datasets/ds003039) ("A walk in the park?"): outdoor walking, steady even-terrain segment | public; `python scripts/setup/00_download_openneuro_dataset.py` | BIDS EEGLAB; gait events from the dataset's events.tsv |

Place the raw data in `datasets/<key>/d00_raw/` (BIDS root). Every later stage
writes to `datasets/<key>/d01_...` to `d05_...` and `d00_qc/`; `datasets/` is
not tracked. Overground caveats: three duplicate-recording groups at the
source (handled by the primary cohort) and sub-019 truncated at the source;
see `docs/cohorts.md`.

## Install

Python 3.13 and [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.13
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
uv pip install -e .             # runtime dependencies, exact versions (pyproject.toml)
uv pip install -e ".[dev]"      # + pytest, ruff
```

`uv.lock` records the full resolved environment. `docs/environment_used.txt`
is the `uv pip freeze` of the environment that produced every stored result:
the direct dependencies equal the pins in `pyproject.toml`; some transitive
packages are older than in `uv.lock`. Stage 5 rerun for S1 and sub-001 in an
environment built from `uv.lock` gives bit-identical ERSP arrays (2026-10-16).

## How to run

The active dataset is set with `BETAGAIT_DATASET=stepup` (default) or
`BETAGAIT_DATASET=jacobsen`. Other environment variables: `BETAGAIT_USE_ASR`
overrides the reference ASR setting (default per dataset: treadmill off,
overground SD 20); `BETAGAIT_DATASETS_DIR` points all data inputs and outputs
to another folder (used to rerun a stage for one subject without touching
`datasets/`).

```bash
# 1. Reference pipeline, per dataset (stages 1-7; prepana02-05 accept one subject ID)
BETAGAIT_DATASET=stepup   python scripts/main_script_eegpipeline.py
BETAGAIT_DATASET=jacobsen python scripts/main_script_eegpipeline.py

# 2. Supplementary reference figures
python scripts/prepana06b_plot_gpm_supplement.py                        # overground only
BETAGAIT_DATASET=<key> python scripts/prepana06c_plot_roi_filter_supplement.py

# 3. Multiverse, per dataset
BETAGAIT_DATASET=<key> python scripts/main_script_multiverse.py         # mulana01-03 (--force restarts)
BETAGAIT_DATASET=<key> python scripts/mulana04_zoom_universes.py
python scripts/mulana05_lowpass40_supplement.py                         # treadmill only

# 4. QC tables and checks (one script per question; see the QC READMEs)
BETAGAIT_DATASET=<key> python scripts/qc/qc_build_tables.py
BETAGAIT_DATASET=<key> python scripts/qc/qc_applied_settings.py
BETAGAIT_DATASET=<key> python scripts/qc/qc_baseline_window.py
BETAGAIT_DATASET=<key> python scripts/qc/qc_universe_subject_contrasts.py
python scripts/qc/qc_jacobsen_duplicate_audit.py                        # overground cohorts
python scripts/qc/qc_averaging_order.py [stepup]                        # default overground
python scripts/qc/qc_log_averaging_bias.py
python scripts/qc/qc_gpm_band_check.py
BETAGAIT_DATASET=<key> python scripts/qc/qc_roi_channel_sensitivity.py  # + supplementary ROI table, topography
BETAGAIT_DATASET=<key> python scripts/qc/qc_ica_audit.py
BETAGAIT_DATASET=<key> python scripts/qc/qc_heelstrike_locked.py
```

`scripts/freeze_gait_anchors.py` recomputes the frozen gait-event anchors and
is run by hand only (it asks before overwriting).

**Runtimes** (one Windows workstation): treadmill stage 2 about 5 min per
subject; overground stage 2 1.5-2 h per subject (run `prepana02 <subject>`,
one subject per process); stages 3-5 a few minutes per subject; multiverse
about 16 h treadmill, longer overground; `mulana04` 40-45 min per dataset on
the first run, seconds afterwards (cached panel arrays);
`qc_averaging_order.py` about 25 min per dataset; `qc_log_averaging_bias.py` about
10 min; everything else under a minute.

**Determinism:** AutoReject and ICA use `RANDOM_STATE = 42`; gait-event
anchors are frozen files. Rerunning stages 1-5 for one subject reproduces the
stored outputs bit for bit, and regenerating any figure from the stored data
gives a byte-identical PNG. Intermediate `.fif` files are float32 (MNE
default); see `docs/decisions.md`.

## Stage map

| Script | Stage |
|---|---|
| `prepana01_prep_gaitevents.py` | 1. Gait events (motion capture or events.tsv) to gait cycles; cycle tables |
| `prepana02_raw2ica.py` | 2. Raw EEG to band-passed, (ASR-corrected,) average-referenced data; AutoReject; ICA fit |
| `prepana03_ica2clean.py` | 3. ICLabel classification; ICA applied to the preprocessed data |
| `prepana04_clean2gaitcycles.py` | 4. ICA-corrected data cut into gait-cycle segments; kept-cycle table |
| `prepana05_gaitcycles2tfr.py` | 5. Per-cycle Morlet TFR, standing baseline, time warp; per-subject ERSP and ROI weights |
| `prepana06_plotbetagait.py` | 6. Group ERSP figure (main text); `06b`: overground GPM supplement; `06c`: ROI-weight supplement |
| `prepana07_betaphase_stats.py` | 7. Group double-stance vs swing paired t-test (reference statistic) |
| `mulana01_create_multiverse.py` | M1. Write the 27 universe scripts (COMET) |
| `mulana02_run_multiverse.py` | M2. Run every universe per subject (ICA per branch, cached) |
| `mulana03_visualize_multiverse.py` | M3. Specification curve per dataset |
| `mulana04_zoom_universes.py` | M4. Lowest / median / highest universe figure |
| `mulana05_lowpass40_supplement.py` | M5. Treadmill 40 Hz vs 60 Hz low-pass multiverse (supplement table and curve) |

## Repository map

```
scripts/           one script per step; header = purpose, inputs, outputs, manuscript section, run
  prepana01-07_*.py      reference pipeline stages (gait events, preprocessing + ICA, ICLabel,
                         gait-cycle segments, TFR + ERSP, group figure, group statistic)
  prepana06b/06c_*.py    supplementary figures (GPM, ROI weights)
  mulana01-05_*.py       multiverse (create, run, specification curve, zoom figure, 40 Hz supplement)
  main_script_*.py       run the reference pipeline / the multiverse in order
  freeze_gait_anchors.py by hand only
  qc/                    QC scripts whose outputs are reported
  setup/                 dataset download
  diag/                  local diagnostics (not tracked)
src/               analysis code imported by the scripts (module table below)
tests/             pytest, synthetic data only (< 1 min)
docs/              decisions.md, figures.md, cohorts.md, parameters.md, environment_used.txt, literature/
results/
  figures/                      main-text figures
  supplementary/                supplementary figures and the 40 Hz table
  pipeline/<key>/qc/            statistics and QC tables (README inside)
  multiverse/<key>/outputs/     per-universe, per-subject table; zoom cache (tracked)
  multiverse/<key>/comet/, branches/   COMET state and ICA caches (not tracked)
  multiverse/stepup_40hz_archive/      earlier 40 Hz treadmill run (read-only, not tracked)
datasets/          raw and processed data (not tracked)
logs/              run logs (not tracked)
```

## Modules in src/

| Module | Purpose | Used by |
|---|---|---|
| `config.py` | Every shared parameter, grouped by Methods section; `python -m src.config [--markdown]` | all |
| `config_stepup.py`, `config_jacobsen.py` | Per-dataset settings (subjects, filters, ASR, events) | `config.py` |
| `paths.py` | Per-dataset data folders (`get_dataset_dirs`) | all stages |
| `resume.py`, `stage_runner.py` | Skip a stage whose outputs are newer than its inputs; subject loop and command line | prepana01-05 |
| `pipeline_steps.py` | Load and annotate raw data, filter, ASR, re-reference, AutoReject, ICA fit and application | prepana02-03, multiverse |
| `preprocessing.py` | Drop channels without valid data | `pipeline_steps.py` |
| `nodes/asr_node.py`, `nodes/gedai_node.py` | ASR decision node; GEDAI node (evaluated, not used) | `pipeline_steps.py` |
| `ica_utils.py` | ICLabel rules and IC topography plots | prepana02-03, multiverse |
| `applied_settings.py` | Check that the saved files carry the configured settings | prepana02, `qc_applied_settings.py` |
| `gait_events.py` | Gait events per subject (motion capture or events.tsv) | prepana01 |
| `gait_cycles.py` | Gait-event detection, cycle extraction and gait QC | prepana01, `gait_events.py` |
| `gait_segments.py` | Cut and save gait-cycle EEG segments | prepana04 |
| `cycle_tfr.py` | Per-cycle Morlet TFR, edge crop, event indices, standing segment | stage 5, multiverse, QC |
| `ersp.py` | Warp anchors, time warp, phase windows, beta band | stage 5-7, multiverse |
| `spatial_filter.py` | Linear Cz-ROI weights | stage 5-7, multiverse |
| `reference_ersp.py` | Per-subject reference ERSP: baseline, warp, phase averages, QC | prepana05 |
| `reference_stats.py` | Per-subject phase beta values; paired t-test | prepana07, `duplicate_audit.py` |
| `reference_figure.py` | Group ERSP figure | prepana06 |
| `gpm.py` | Gait power modulation (cycle mean removed) | prepana06b, QC |
| `multiverse_pipeline.py` | One subject through one multiverse branch; cached universe ERSP | universe scripts (mulana01-02), `multiverse_zoom.py` |
| `multiverse_results.py` | Read the combined COMET results; per-universe statistics | mulana03-05, QC |
| `spec_curve.py` | Specification-curve figure | mulana03, mulana05 |
| `multiverse_zoom.py` | Lowest / median / highest universe figure | mulana04 |
| `qc.py` | Per-subject QC records, QC flags, overground cohort loaders | all stages, QC |
| `qc_tables.py` | Reviewer QC tables of one dataset | `qc_build_tables.py` |
| `duplicate_audit.py` | Overground raw-data and duplicate-recording audit | `qc_jacobsen_duplicate_audit.py` |
| `averaging_order.py` | ERSP under our vs the source study's averaging order | `qc_averaging_order.py`, `qc_log_averaging_bias.py` |
| `roi_diagnostics.py` | Channel-level beta contrast, ROI sensitivity variants, ICA back-projection, heel-strike-locked averages | `qc_roi_channel_sensitivity.py`, `qc_ica_audit.py`, `qc_heelstrike_locked.py` |
| `roi_figures.py` | Contrast topography, frequency profile and heel-strike plots | the same three QC scripts |
| `plot_style.py` | Shared figure style | all figures |

## Where things are

- **Parameters:** `src/config.py` (shared, grouped by Methods section) and
  `src/config_stepup.py` / `src/config_jacobsen.py`; scripts read everything
  from `src.config`. `python -m src.config` prints both datasets side by side;
  `docs/parameters.md` is generated with `python -m src.config --markdown`.
- **Decisions and reasons, differences from the source study:** `docs/decisions.md`.
- **Figures** (file, script, n, colour range, captions): `docs/figures.md`.
- **Cohorts:** `docs/cohorts.md`.
- **QC and statistics:** `results/pipeline/<key>/qc/README.md` (file, script,
  manuscript section). Per-subject QC jsons stay in `datasets/<key>/d00_qc/`.

## Tests and style

```bash
python -m pytest            # synthetic data, < 1 min
ruff check . && ruff format --check .
```

Ruff rules E, F, I, B, UP, N and D (numpy docstrings) are enabled.

## Limitations

- Overground: turns within the outdoor route cannot be detected or excluded;
  the effective number of independent recordings is 14 to 16 (duplicates at
  the source).
- The overground absolute ERD level depends on the averaging order; the
  double-stance - swing contrast does not (`docs/decisions.md`).
- Bad channels interpolated per subject are printed but not stored.
- Two treadmill exclusion checks (S21, S2) were run with scripts that were not
  preserved; their outputs are kept and cited in `stepup_exclusions.tsv`.
- GEDAI was evaluated as a candidate node and excluded (`docs/decisions.md`).
  Its evidence (`stepup_gedai_gate_check.txt` and `scripts/diag/`) is kept
  locally and is not part of the repository.

## Open items

- Data-access statement for the treadmill (stepUpAms) dataset.
- No LICENSE or CITATION.cff yet.

## References

- Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science*, 11(5), 702-712.
- Clayson PE, Carbine KA, Baldwin SA, Larson MJ (2021). Methodological reporting behavior, sample sizes, and statistical power in studies of event-related potentials. *Psychophysiology*, 58(2), e13437.
- Jacobsen NSJ, Blum S, Witt K, Debener S (2020). A walk in the park? Characterizing gait-related artifacts in mobile EEG recordings. *European Journal of Neuroscience*.
- Klug M, Gramann K (2021). Identifying key factors for improving ICA-based decomposition of EEG data in mobile and stationary experiments. *European Journal of Neuroscience*, 54(12), 8406-8420.
- Delorme A (2023). EEG is better left alone. *Scientific Reports*, 13, 2372.
- Chang C-Y, Hsu S-H, Pion-Tonachini L, Jung T-P (2020). Evaluation of Artifact Subspace Reconstruction for automatic artifact components removal in multi-channel EEG recordings. *IEEE Transactions on Biomedical Engineering*, 67(4), 1114-1121.
- Mullen TR, et al. (2015). Real-time neuroimaging and cognitive monitoring using wearable dry EEG. *IEEE Transactions on Biomedical Engineering*, 62(11), 2553-2567.
- Pion-Tonachini L, Kreutz-Delgado K, Makeig S (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and feature benchmark. *NeuroImage*, 198, 181-197.
- Grandchamp R, Delorme A (2011). Single-trial normalization for event-related spectral decomposition reduces sensitivity to noisy trials. *Frontiers in Psychology*, 2, 236.
- Gwin JT, Gramann K, Makeig S, Ferris DP (2011). Electrocortical activity is coupled to gait cycle phase during treadmill walking. *NeuroImage*, 54(2), 1289-1296.
