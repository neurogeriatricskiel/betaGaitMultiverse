# Figures

Every figure in `results/figures/` (main text) and `results/supplementary/`,
the script that draws it, its n and its fixed colour range, followed by the
captions. Regenerating a figure from the stored data gives a byte-identical
PNG. Treadmill = stepUpAms (key `stepup`); overground = Jacobsen et al. 2020,
OpenNeuro ds003039 (key `jacobsen`).

| File | Script | n | Colour range (clipped) | Axis limits |
|---|---|---|---|---|
| `figures/stepup_betaersp_gait.png` | `prepana06_plotbetagait.py` | 17 subjects | heatmap ±3 dB (1.70 %); topographies ±6 dB (2.60 %, see below) | beta trace -4.5 to 2.5 dB |
| `figures/jacobsen_betaersp_gait.png` | `prepana06_plotbetagait.py` | 14 subjects (primary) | heatmap ±3 dB (0.47 %); topographies ±3 dB (1.04 %) | beta trace -3.5 to 0.5 dB |
| `figures/stepup_speccurve.png` | `mulana03_visualize_multiverse.py` | 15 subjects per universe | - | -0.5 to 3.5 dB |
| `figures/jacobsen_speccurve.png` | `mulana03_visualize_multiverse.py` | 14 subjects per universe (primary) | - | -0.25 to 1.25 dB |
| `figures/stepup_multiverse_zoom.png` | `mulana04_zoom_universes.py` | 15 subjects | ±5 dB (0.06 %) | beta trace -4.5 to 2.5 dB |
| `figures/jacobsen_multiverse_zoom.png` | `mulana04_zoom_universes.py` | 14 subjects (primary) | ±3.5 dB (0.47 %) | beta trace -3.5 to 0.5 dB |
| `supplementary/jacobsen_gpm_gait.png` | `prepana06b_plot_gpm_supplement.py` | 14 subjects (primary) | ±1 dB (0.43 %) | GPM trace -0.75 to 0.75 dB |
| `supplementary/stepup_roi_filter_topo.png` | `prepana06c_plot_roi_filter_supplement.py` | 64 channels | Reds, 0 to max weight | - |
| `supplementary/jacobsen_roi_filter_topo.png` | `prepana06c_plot_roi_filter_supplement.py` | 64 channels | Reds, 0 to max weight | - |
| `supplementary/stepup_speccurve_lowpass40.png` | `mulana05_lowpass40_supplement.py` | 14-15 subjects per universe | - | -0.5 to 3.5 dB |
| `supplementary/stepup_multiverse_lowpass40_vs_60.tsv` (table) | `mulana05_lowpass40_supplement.py` | 27 universes | - | - |
| `supplementary/stepup_contrast_topography.png` | `qc/qc_roi_channel_sensitivity.py` | 17 subjects | ±4.78 dB (largest channel value, unclipped) | - |
| `supplementary/jacobsen_contrast_topography.png` | `qc/qc_roi_channel_sensitivity.py` | 14 subjects (primary) | ±2.05 dB (largest channel value, unclipped) | - |
| `supplementary/<dataset>_roi_sensitivity.tsv` (table) | `qc/qc_roi_channel_sensitivity.py` | 17 / 14 subjects | - | - |
| `multiverse/<dataset>/outputs/<dataset>_multiverse_frontal.tsv` (table) | `qc/qc_multiverse_frontal.py` | 27 universes, 15 / 14 subjects | - | - |

QC plot (not a manuscript figure): `results/pipeline/jacobsen/qc/jacobsen_gpm_averaging_order.png`
(`scripts/qc/qc_averaging_order.py`). Per-subject QC plots are written under
`datasets/<dataset>/` and are not tracked.

## Common style

All figures use `src/plot_style.py`. Beta band = 13-30 Hz. ROI = linear
Cz-centred spatial filter (`src/spatial_filter.py`): weight
`w_i = 1 - d_i / d_max`, normalised to sum 1, where `d_i` is the 3D distance
from channel *i* to Cz. ERSP values are dB relative to a standing
(quiet-stance) baseline in both datasets. Gait events: RHS / LHS = right /
left heel strike, LTO / RTO = left / right toe-off; phase bar: DS = double
stance, LLS / RLS = left / right single stance (swing of the other leg).

ERSP heatmaps and beta topographies use RdBu_r, zero-centred: white = 0 dB =
standing baseline, blue = below baseline (ERD), red = above (ERS). Each figure
has one fixed symmetric colour range shared by all its heatmap panels, a
colourbar labelled "ERSP (dB re standing baseline)" with true dB ticks, and the
range stated under the title. The range is the smallest of ±{2, 2.5, 3, 3.5, 4,
5, 6} dB that clips at most 2 % of the figure's pixels (GPM: ±{0.5, 1, 1.5, 2}
dB, since GPM spans only about ±0.6 dB). The reference figures' three
topographies have their own colour bar ("Beta ERSP (dB re standing
baseline)") and range, chosen by the same rule over their 3 x 64 channel
values (`TOPO_VLIM` in `src/plot_style.py`); both ranges are stated under the
title ("Heatmap scale ±X dB, Topography scale ±Y dB"):

| Figure | Heatmap | Topographies | Topography values clipped at ±2 / 2.5 / 3 / 3.5 / 4 / 5 / 6 dB |
|---|---|---|---|
| Treadmill reference | ±3 dB (1.70 %) | ±6 dB (2.60 %) | 22.40 / 11.46 / 6.77 / 5.73 / 4.69 / 3.65 / 2.60 % |
| Overground reference | ±3 dB (0.47 %) | ±3 dB (1.04 %) | 39.06 / 13.54 / 1.04 / 0 / 0 / 0 / 0 % |

No candidate meets the 2 % rule for the treadmill topographies; ±6 dB, the
largest candidate, is used. The five clipped values are all in the
double-stance map: Fz +7.76, Fpz +6.59, F1 +6.53, F2 +6.39, AF3 +6.17 dB.

p values printed on figures are given as "p < 0.001" when below 0.001
(`plot_style.format_p`); the exact values are in the statistics files. Beta-band lines (13 and 30 Hz, dashed)
and gait-event lines (dotted) are black with a white outline. Beta-trace
y-limits are fixed per dataset and shared by that dataset's reference and zoom
figures; each includes 0 dB as a dashed line. Titles: "<Treadmill|Overground>
walking: <content> (n = N)", with n computed from the data.

## Captions

### Treadmill reference ERSP and topographies
`results/figures/stepup_betaersp_gait.png`

Group-average (n = 17) ERSP, 8-60 Hz, at the Cz linear ROI across the
normalised gait cycle (0-100 %, RHS to RHS). Gait events are marked at the
frozen group anchors the ERSP arrays were warped to (LTO 10.0 %, LHS 50.0 %,
RTO 59.8 %), with the DS / LLS / DS / RLS phase bar beneath. Under the heatmap,
the beta-band (13-30 Hz) ROI trace. Right: beta topographies for the whole
cycle, swing only and double stance only, with their own colour bar. Heatmap
colour range ±3 dB (1.70 % clipped); topography range ±6 dB (2.60 % of
channel values clipped, all frontal in the double-stance map, up to
+7.76 dB at Fz). The treadmill beta ERSP spans both signs: ERS above baseline
around double stance, ERD below baseline during swing.

### Overground reference ERSP and topographies
`results/figures/jacobsen_betaersp_gait.png`

Same construction on the overground primary independent-recordings cohort
(n = 14; drops sub-006 / 007 / 008 / 013, read from
`jacobsen_duplicate_recordings.txt`, the same source as the primary statistic
t(13) = 3.50). 0.2-60 Hz band-pass. Gait events at the fixed warp latencies of
the source study (LTO 18 %, LHS 50 %, RTO 68 %). Heatmap and topographies
each ±3 dB, with separate colour bars. The
overground beta ERSP is almost entirely below the standing baseline, so the
panels are almost entirely blue: the double-stance vs swing effect here is a
modulation of ERD depth, not an ERD / ERS sign change. This absolute level
depends on the averaging order (see `docs/decisions.md`).

### Treadmill specification curve
`results/figures/stepup_speccurve.png`

All 27 universes (high-pass 0.5 / 1.0 / 2.0 Hz x ASR off / SD 3 / SD 20 x IC
selection conservative / balanced / liberal), n = 15 subjects per universe
(`MULTIVERSE_SUBJECTS`, excludes S2, S10, S21; a smaller cohort than the
n = 17 reference cohort). Top: one point per universe, sorted by the
group-mean per-subject beta contrast (double stance - swing, dB); grey band =
t-based 95 % CI across subjects, drawn as in comet's specification curve
(cubic interpolating spline through each universe's bounds): at each universe
it equals the stored CI, between universes it is interpolation only (dips at
most 0.025 dB below, and rises at most 0.037 dB above, the neighbouring
bounds); green = p < 0.05,
one-sample t-test of the per-subject contrasts against 0, uncorrected for the
27 universes (27 / 27 universes). Bottom: decision values of each universe,
rows ordered as in Table 2.1. Fixed y-limits -0.5 to 3.5 dB. Per-subject
values: `results/multiverse/stepup/outputs/stepup_universe_subject_contrasts.tsv`.

### Overground specification curve
`results/figures/jacobsen_speccurve.png`

Same code, size, grid and construction on the overground primary cohort
(n = 14; the pkl's `dedup_diffs` field). 27 / 27 universes p < 0.05,
uncorrected. Smoothed CI band as above (between universes within 0.009 dB of
the neighbouring bounds). Fixed y-limits -0.25 to 1.25 dB. Per-subject values:
`results/multiverse/jacobsen/outputs/jacobsen_universe_subject_contrasts.tsv`.

### Treadmill multiverse zoom: lowest / median / highest universe
`results/figures/stepup_multiverse_zoom.png`

Group-average ROI ERSP (8-40 Hz, the multiverse TFR range) for three of the 27
treadmill universes, n = 15: the lowest-, median- (14th of 27) and
highest-ranked by group-mean double-stance - swing beta contrast. Lowest:
HP 2.0 Hz, ASR SD 3, IC conservative (+0.29 dB, t(14) = 3.34, p = 0.00483);
median: HP 0.5 Hz, ASR SD 20, IC conservative (+1.46 dB, t(14) = 6.05,
p < 0.001); highest: HP 0.5 Hz, ASR off, IC balanced (+2.05 dB,
t(14) = 5.16, p < 0.001; exact p 2.99e-05 and 1.45e-04 in
`stepup_multiverse_universes.tsv`). Under each heatmap, the beta (13-30 Hz) ROI trace
on a shared y-axis (-4.5 to 2.5 dB). One colour range (±5 dB). Gait events at
the frozen group anchors used by the multiverse (LTO 10.0 %, LHS 50.0 %,
RTO 59.8 %). Panel arrays are cached in
`results/multiverse/stepup/outputs/stepup_zoom_arrays.npz`.

### Overground multiverse zoom: lowest / median / highest universe
`results/figures/jacobsen_multiverse_zoom.png`

Same construction on the overground primary cohort (n = 14). Lowest:
HP 1.0 Hz, ASR SD 20, IC conservative (+0.22 dB, t(13) = 2.68, p = 0.0189);
median: HP 1.0 Hz, ASR SD 3, IC liberal (+0.39 dB, t(13) = 3.95,
p = 0.00165); highest: HP 0.5 Hz, ASR off, IC liberal (+0.68 dB,
t(13) = 4.06, p = 0.00134). Values equal the specification curve's. Colour
range ±3.5 dB; beta-trace y-limits -3.5 to 0.5 dB. Gait events at the frozen
group anchors used by the multiverse (LTO 16.0 %, LHS 50.3 %, RTO 66.0 %),
which differ from the fixed 18 / 50 / 68 % of the overground reference figure.
Panel arrays are cached in
`results/multiverse/jacobsen/outputs/jacobsen_zoom_arrays.npz`.

### Supplementary: overground gait power modulation (GPM)
`results/supplementary/jacobsen_gpm_gait.png`

Gait power modulation as defined in the source study (Jacobsen et al.), for
comparison with their GPM panel: GPM(f, t) = ERSP(f, t) - mean_t ERSP(f, t),
the standing-baselined ERSP minus its own mean over the gait cycle at each
frequency, per subject and channel, then averaged across subjects (n = 14,
primary cohort). Each frequency row averages to 0 dB over the cycle, so GPM
shows only the within-cycle modulation, not the overall ERD level. The
double-stance - swing contrast is unchanged by the subtraction (recomputed
from the GPM arrays: t(13) = 3.501, p = 0.0039, +0.582 dB, identical to the
primary result). Left: linear ROI centred on Cz; right: single channel Cz, as
in the source. RdBu_r zero-centred, white = 0 dB = cycle mean, colourbar
"GPM (dB)", one shared range ±1 dB. Beta-band GPM trace under each heatmap
(y-limits -0.75 to 0.75 dB). Gait events at the fixed warp latencies
(18 / 50 / 68 %). 8-60 Hz; the source panels start at 2 Hz but our TFR starts
at 8 Hz; the narrow 50 Hz row reflects the line-noise notch filter. The source
pipeline's spectral PCA step is not applied.

### Supplementary: ROI spatial-filter weight topographies
`results/supplementary/stepup_roi_filter_topo.png`,
`results/supplementary/jacobsen_roi_filter_topo.png`

Methods figure (not a result): the linear Cz-centred ROI weight applied to
every channel before the beta-ERSP reduction; weights fall linearly with
distance from Cz and reach 0 at the farthest channel. Sequential "Reds"
(weights are non-negative), colourbar "ROI weight". Here n counts channels:
the weights are identical for every subject of a dataset (same positions) but
differ between datasets because the position sources differ.
Treadmill: positions from the standard_1005 template; 64 EEG channels (the
STATUS and "Counter 2power24" trigger channels, typed EEG in channels.tsv, are
dropped on loading). M1 and M2 have template positions and enter the average
reference and the ICA; M1 is the farthest channel from Cz (19.3 cm) and gets
weight 0, M2 (19.2 cm) 0.00008. Highest weights: Cz 0.036, then C1, CPz, FCz,
C2 (0.029, about 3.8 cm).
Overground: positions from each recording's electrodes.tsv; 64 EEG channels
(3 accelerometer channels dropped). FT9, TP9, FT10 and TP10 are the farthest
channels from Cz (15.9 cm, tied) and get weight 0. Highest weights: Cz 0.054,
then CPz 0.042, C1 and C2 0.041.

### Supplementary: treadmill multiverse with a 40 Hz low-pass
`results/supplementary/stepup_speccurve_lowpass40.png`,
`results/supplementary/stepup_multiverse_lowpass40_vs_60.tsv`

The treadmill multiverse was first run with a 40 Hz raw low-pass (archived,
read-only, `results/multiverse/stepup_40hz_archive/`); the reported run uses
60 Hz. Same 27-universe grid and per-subject contrast. The specification curve
is drawn with the main figure's code (smoothed CI band; between universes
within 0.021 dB below and 0.009 dB above the neighbouring bounds) and y-limits
(-0.5 to 3.5 dB); n = 15 in
26 universes and 14 in universe 6 (HP 0.5 Hz, ASR SD 3, IC liberal). The table
gives, per universe, the mean contrast (dB), t, p and n at 60 Hz and at 40 Hz;
its 60 Hz columns equal the main specification curve.

### Supplementary: beta contrast per channel
`results/supplementary/stepup_contrast_topography.png`,
`results/supplementary/jacobsen_contrast_topography.png`

Group-mean double stance minus swing beta (13-30 Hz) ERSP at every channel,
from the stored per-subject phase maps of the reference cohort (treadmill
n = 17, overground primary n = 14); the linear ROI applied to these channel
values gives the reference statistic exactly. RdBu_r, white = 0 dB; the
colour limit is the largest absolute channel value, so nothing is clipped
(treadmill ±4.78 dB, overground ±2.05 dB, stated on the figure). The map is
interpolated only within the electrode hull (`extrapolate="local"`).
Treadmill: positive at every channel, largest frontally (Fz +4.78 dB, then
F1, F2, Fpz, AF3, AF4, all above +4.1 dB) and smallest over central
channels (C3 to C4, CPz: +0.64 to +1.09 dB; Cz +0.73 dB). Overground: largest at occipital and inferior channels (I1 +2.05 dB,
PO9, PO10, I2); central channels +0.44 to +0.67 dB (Cz +0.64 dB). Per-channel values, t and
contributions to the ROI contrast: `results/pipeline/<dataset>/qc/<dataset>_channel_contrast.tsv`.

### Supplementary table: ROI sensitivity
`results/supplementary/stepup_roi_sensitivity.tsv`,
`results/supplementary/jacobsen_roi_sensitivity.tsv`

Paired double stance vs swing test (one-sample t of the per-subject ROI
contrast; t, df, p, mean dB) for four ROIs, weights renormalised to sum 1:
the stored linear Cz ROI; without Fp*/AF*; without Fp*/AF*, F7/F8, FT*, T*
and TP*; and the five channels nearest Cz with equal weights (treadmill Cz,
C1, CPz, FCz, C2; overground Cz, CPz, C1, C2, FC2). `n_channels` counts the
channels with non-zero weight (the farthest channels already have weight 0).
Treadmill t(16) = 5.13 / 5.02 / 5.05 / 4.42 (+1.95 / +1.82 / +1.81 /
+1.15 dB); overground t(13) = 3.50 / 3.51 / 3.49 / 3.44 (+0.58 / +0.57 /
+0.56 / +0.56 dB).

### Table: frontal signal and ROI variants across the multiverse
`results/multiverse/<dataset>/outputs/<dataset>_multiverse_frontal.tsv`

One row per universe (27; treadmill n = 15, overground primary n = 14),
sorted by the stored contrast: high-pass, ASR, IC rule; group heel-strike-
locked peak-to-peak deflection (1-40 Hz, 0-400 ms after heel strike, baseline
-200 to 0 ms) at Fpz and Cz; double stance vs swing beta contrast (dB, t, p)
with the linear Cz ROI, without Fp*/AF*, and for the five channels nearest Cz
(equal weights); and the stored contrast. Computed from each universe's
float32 ICA-corrected cache, whereas the stored results used float64 data in
memory, so per-subject contrasts differ from the stored ones by up to
0.062 dB (treadmill; stated in the file header) and universe means by about
0.001 dB.
