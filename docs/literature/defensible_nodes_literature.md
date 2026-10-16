# Preprocessing Decision-Node Defensibility — Detailed Rationale

**Literature review:** 65 mobile-EEG gait/gait-adjacent papers, listed in the review (Vinod V, et al., 2026) with respect to the preprocessing parameters used - **`master_literature_reference.csv`** (same folder). Every count and quote below can be traced back to a row in that file via the `Filename` column.

---
## Defensible nodes in the multiverse

## 1. ASR on / off

- **Literature:** 18/65 papers (~28%) apply Artifact Subspace Reconstruction. This is a genuine minority-use technique, not a near-universal step — the opposite situation from something like "use ICA," where equipoise doesn't exist.
- **Why it's contested, not just uncommon:** ASR use does not track a single study design or population. It appears across healthy-adult and Parkinson's-disease cohorts, treadmill and overground protocols, and low- and high-channel-count montages (8 to 264 electrodes in the corpus). There is no obvious confound (e.g., "only used for high-channel EGI systems") that would make on/off a proxy for something else. That means the on/off decision is a free methodological choice made independently by different labs — the definition of a forking path.
- **Reported cutoffs vary widely and inconsistently:** where a cutoff is stated, values include 3, 5, 7, 10, 12, 15, 20 SD, with two papers (Tortora et al., 2020; Pellegrini et al., 2024) reporting ASR was used but not stating a value at all. This spread (3–20 SD, a >6-fold range) with no dominant mode is itself evidence that the field has not converged on ASR parameterization — it has barely converged on whether to use ASR at all.
- **Why on/off (not cutoff-value) is the node:** Because only about a third of ASR-using papers report a cutoff, treating cutoff value as a multiverse node would mean building an axis where two-thirds of the "informing" literature contributes nothing. A binary on/off axis, by contrast, is fully informed by all 65 papers (each one unambiguously did or did not report ASR). For the fixed ASR arm, use the classic Chang/Mullen-recommended value of 20 SD, which is both the most conservative end of the observed range and the value explicitly endorsed as a "lax" (brain-signal-preserving) setting by Artoni et al. (2017) in this corpus.
- **Caveat for the manuscript:** note explicitly that "ASR on" in the multiverse uses a single fixed cutoff (20 SD) and therefore only tests the presence/absence of the correction step, not the sensitivity of the correction to its own tuning parameter. This should be flagged as a scope limitation, not silently absorbed into the ASR-on arm.

---

## 2. IC-selection method (ICLabel vs. dipole/residual-variance vs. manual)

- **Literature:** of the 65 papers, roughly 24 use manual/visual component rejection, 24 use a dipole-fit/residual-variance criterion, and 15 use ICLabel (with partial overlap — some papers report both a dipole-RV criterion and an ICLabel probability threshold as joint inclusion criteria, e.g., Jacobsen et al., 2022; De Sanctis et al., 2023).
- **Relevance as a decision node:** no single method reaches even 40% adoption, and adoption is not obviously time-trending — ICLabel papers (available from 2019 onward) coexist with dipole-RV and manual papers throughout 2019–2025, rather than displacing them. That three-way, non-converging split, on the single question "how do you decide an IC is brain vs. artifact," is exactly the axis a methods review flagged as unresolved ("only brain components?").
- **Why manual/visual is excluded from the automated multiverse arm despite being the single largest category:** manual rejection is not a parameterizable, reproducible decision — it cannot be instantiated as a rule that a pipeline can apply identically across universes. Including it would collapse the multiverse logic (every "manual" universe would in practice be one arbitrary set of expert judgment calls, not a systematically varied condition). It is retained in the review's narrative/summary but excluded from the executable multiverse.
- **Why ICLabel vs. dipole-RV specifically:** these are the two automated, reproducible approaches in the corpus with enough independent adoption to be treated as real alternatives rather than one dominant default with a token comparator (15 and 24 papers respectively — both well represented). Implementation: ICLabel brain-probability threshold (see Node 3) vs. dipole residual-variance <15% (the modal RV cutoff wherever RV is reported in the corpus, e.g., Bradford et al., 2016; Nordin et al., 2019; Wagner et al., 2022; De Pauw et al., 2019).
- **Interaction to flag:** several papers use both criteria jointly (dipole location/RV *and* an ICLabel-style probability filter) as a conjunctive rule, which suggests a fourth possible arm ("both criteria required") worth mentioning as a sensitivity check even if not built into the core 2×2×2 design.

---

## 3. ICLabel brain-probability threshold (0.7 vs. 0.9 / equivalently, reject-if-artifact-probability 85–90%)

**Verdict: INCLUDE, nested within the ICLabel arm of Node 2.**

- **Corpus support:** of the 15 ICLabel-using papers, roughly 7 explicitly state a numeric threshold. This is a much higher within-method reporting rate than ASR's cutoff-reporting rate (7/15 ≈ 47% vs. ASR's ~33%), which is why this threshold clears the bar for inclusion as a node while the ASR cutoff value does not.
- **Range and clustering:** reported values fall into two natural families:
  - **"Reject if artifact-probability ≥ 85–90%"** — e.g., Jacobsen et al. (2020): "components exceeding a 90% probability of being eye, muscle, heart, line noise, and channel noise were rejected"; Jacobsen et al. (2022) uses the same 90% eye/blink criterion; Mustile et al. (2022) and Mustile et al. (2023) use 90% across the same five artifact classes; Dutke et al. (2025) uses 85%.
  - **"Keep if brain-probability ≥ 70%"** — e.g., Scanlon et al. (2022): ICs "classified with less than 70% probability of being due to brain activity" were removed; Stuart et al. (2021) requires ">70% likelihood of being brain activity."
  - A third, looser convention (brain-probability >50%, i.e., "more likely brain than not") appears in Protzak et al. (2021), Patelaki et al. (2022/2023/2024 papers), and the Jacobsen et al. (2024) theta-adaptation paper — effectively a majority-vote rather than a conservative-exclusion rule.
- **Why 0.7 and 0.9 specifically bracket the defensible range:** the 90%-artifact-probability convention and the 70%-brain-probability convention are algebraically close but not identical (rejecting at ≥90% artifact-probability is stricter than keeping at ≥70% brain-probability, since an IC could have 75% brain / 25% artifact probability and pass the second rule but not fail the first). Using both as separate arms lets the multiverse span from a lenient default (50%, used by nearly a third of ICLabel papers, but arguably too permissive to serve as a bracketing extreme) up through the two most common stricter conventions in the literature. Recommend 0.7 (lenient-strict) and 0.9 (strict) as the two levels, explicitly noting the 50% convention as a "field-common but not-selected-for-multiverse" comparator in the write-up, since it is closer to a default-software-setting than a deliberate methodological choice.
- **Reported vs. unreported ICLabel users:** the remaining ~8 ICLabel papers describe using the classifier as *part* of a composite rejection rule (with RV, dipole location, spectral slope, etc.) without stating the isolated ICLabel probability cutoff (e.g., De Sanctis et al., 2023; Salminen et al., 2025). These are retained in the IC-selection-method count (Node 2) but do not contribute a data point to this threshold node — this is exactly analogous to the ASR situation and is the reason both nodes state the "reporting completeness" caveat.

---

## 4. Baseline type for gait-ERSP normalization (gait-cycle mean vs. standing/condition reference)

**Verdict: INCLUDE, but reframe as an invariance demonstration rather than a source of expected effect variance.**

- **Corpus support:** gait-cycle mean baselining is by far the dominant convention (18 papers), with condition/standing reference used in only 2 (Nordin et al., 2019 dual-layer paper; Vitório et al., 2023). A further 25 papers apply *some* baseline but do not specify the reference window precisely enough to classify ("applied, unspecified"), and 10 report no baseline step at all.
- **Both levels are attested in the field**, so it is legitimate to include baseline type as a multiverse arm on adoption grounds alone.
- **Why it should be reframed rather than treated as a normal effect-generating node:** for a *paired, within-cycle* contrast (e.g., double-support vs. swing phase, both measured within the same gait cycle and same baseline), a baseline that is constant across the cycle — whether it's a standing recording or the whole-cycle mean — enters the contrast as an additive constant and cancels algebraically. This means the two baseline choices are mathematically expected to be inert for exactly the comparisons this multiverse is built around, even though both are "real" choices in the literature. Recommendation: keep both levels in the design, but present the (near-)invariance of the paired contrast to baseline choice as a positive validation result in the manuscript — evidence that the pipeline's downstream comparison is robust to this particular forking path — rather than reporting it alongside ASR/IC-selection as if it were expected to move the estimate.

---

## 5. High-pass filter cutoff (1 Hz vs. 0.1 Hz, or 1 Hz vs. 2 Hz)

**Verdict: CANDIDATE / optional, not part of the core design.**

- **Corpus support:** 1 Hz is clearly modal (≈20 papers state it explicitly), while 0.1 Hz appears in only 2 (Luu et al., 2017; Protzak et al., 2021). A further ~2 papers use other values (0.75 Hz: Wagner et al., 2022, Dutke et al., 2025; 2 Hz: De Sanctis et al., 2023; 20 Hz — an unusually aggressive value — Roeder et al., 2020/2024). The remainder do not report a high-pass value at all.
- **Why it's a weak node on adoption grounds:** a 19-vs-2 split is not equipoise — it reflects a near-consensus (1 Hz) with a couple of outlier choices, not two genuinely competing conventions. Treating it as a full multiverse node risks implying the field is split roughly down the middle, which it is not.
- **Why it may still be worth an optional sensitivity check:** the *reason* labs deviate from 1 Hz is methodologically principled rather than arbitrary — lower cutoffs (0.1 Hz) preserve slow cortical potentials at the cost of noisier ICA decomposition, while cutoffs at or above 1 Hz measurably improve ICA quality in high-channel-count, high-mobility recordings (this is a documented benchmarked trade-off in the methods literature — Klug & Gramann, 2021, EJN, is the key reference: high-pass cutoff up to 2 Hz improves ICA quality as channel count and movement increase). Because this trade-off is well characterized outside the gait corpus itself, a 1-vs-2 Hz (rather than 1-vs-0.1 Hz) contrast is the more literature-grounded version of this node if it is included — 2 Hz is a documented upper bound for ICA-quality benefit, whereas 0.1 Hz is a minority choice with only 2 supporting papers in this specific corpus.
- **Recommendation:** fix at 1 Hz as the backbone constant for the core 2×2×2 design; include the 1-vs-2 Hz contrast only as a named, separate ICA-stability sensitivity analysis, explicitly framed as testing preprocessing-quality trade-offs rather than as a fourth arm of the main multiverse (which would make it a 2×2×2×2 = 16-universe design and dilute interpretability).

---

## 6. GEDAI (excluded from main design)

**Verdict: EXCLUDE from the multiverse; report only as a methods-validation sub-study.**

- **Corpus support:** 0/65 papers in this gait corpus use GEDAI — there is no adoption precedent to justify treating it as a forking path the field actually takes.
- **Empirical failure on this dataset:** in preliminary testing, GEDAI produced a leadfield/10-05 montage mismatch and removed an implausibly large fraction of signal variance (~97% overall, including 25–95% of Cz-ROI beta power, with a fitted threshold of 0.000 in all bands) — a clear sign of pathological rather than selective artifact rejection on this montage/dataset combination.
- **Why exclude rather than "include as a level that performs badly":** a multiverse node is meant to represent a genuine, defensible choice a careful analyst might make; a method with zero precedent in the target literature and a demonstrated catastrophic failure mode on the actual data does not meet that bar. Reporting it as a cautionary methods-validation note (rather than silently omitting it, or including it as if it were a legitimate arm) preserves the transparency value of having tried it without distorting the main multiverse's interpretability.

---

## 7. Fixed constants (no equivalent node — apply identically across every universe)

Low-pass filtering (~40 Hz), line-noise removal (notch), average referencing, and the presence of an ICA decomposition step are treated as fixed backbone steps rather than nodes because adoption is effectively universal in the corpus (ICA: 65/65; the others near-ubiquitous wherever reported), so there is no field-level equipoise to model. These are reported explicitly as the fixed backbone of every universe rather than omitted, so that the multiverse's scope (which choices were varied vs. held constant) is fully transparent to a reader.

---

## Recommended core multiverse design (restated with rationale summary)

| Node | Levels | Basis for inclusion |
|---|---|---|
| 1. ASR | on (20 SD) / off | 18/65 use it; on/off is a genuine, fully-informed binary; cutoff value is not (only ~1/3 of users report it) |
| 2. IC-selection method | ICLabel (probability) / dipole-RV (<15%) | Three-way field split (24/24/15) with no majority; these two are the reproducible, automatable arms |
| 3. ICLabel brain threshold | 0.7 (brain-probability) / 0.9 (artifact-probability-reject) | Both conventions independently attested (7/15 ICLabel papers report a threshold); brackets the reported range |

Plus: **baseline type** (gait-cycle mean / standing-condition reference) retained as an invariance demonstration, not counted toward expected effect variance for paired within-cycle contrasts.

Fixed across all universes: high-pass 1 Hz, low-pass ~40 Hz, notch filter, average reference, Extended-Infomax ICA present.

Excluded: GEDAI (no corpus precedent + empirical failure on this dataset) → reported as a separate methods-validation note.

Optional add-on (not part of the core 8-universe design): high-pass 1 vs. 2 Hz, as a named ICA-stability sensitivity check, grounded in Klug & Gramann (2021) rather than in this corpus's own (weak, 19-vs-2) equipoise.

---

## Additional gait-specific / MNE-implementable nodes considered (not yet scored against this corpus's adoption counts, but literature-grounded)

These extend beyond the ASR / IC-selection / baseline core and were flagged from benchmarking and gait-specific methods papers rather than raw corpus usage frequency. They are listed here for completeness; each would need its own adoption tally against the 65-paper corpus before being promoted to "strong" status the way Nodes 1–3 above have been.

- **Bad-channel rejection criterion** (clean_rawdata-style correlation/SD threshold vs. interpolation-only) — STRONG candidate per Delorme (2023, *Sci Rep*), which found bad-channel handling to be one of only two preprocessing steps that reliably help across four major toolboxes (the other being high-pass filtering).
- **ICA algorithm** (Extended Infomax vs. AMICA/Picard) — CANDIDATE; AMICA and Extended Infomax dominate this corpus, but reported effect sizes for algorithm choice are typically modest (cf. Leutheuser et al., 2013, on AMICA vs. Infomax for EMG removal).
- **Line-noise removal method** (notch vs. Zapline/ZapLine-plus) — CANDIDATE; a clean methodological contrast (de Cheveigne, 2020) with modest expected impact on beta-band results specifically.
- **Time-warping of the gait cycle** (endpoint-only normalization vs. warping intermediate gait events to group-median latencies) — GAIT-SPECIFIC node; the field standard (Gwin et al., 2011; Bradford et al., 2016; Nordin et al., 2019) warps intermediate events (heel-strike/toe-off), not just cycle endpoints, so this is a defensible axis if the current pipeline only does endpoint normalization.
- **Epoch/cycle rejection threshold** (fixed amplitude, e.g. 3 SD, vs. data-driven autoreject) — CANDIDATE; both conventions appear in the broader mobile-EEG literature (Jas et al., 2017, for autoreject; Gwin et al., 2011, for a fixed 3 SD gait-event threshold).
- **Reference scheme** (average vs. REST/robust average) — CANDIDATE (weak); flagged with caution because Delorme (2023) found that some "improved" referencing schemes can *reduce* ERP-analysis reliability relative to plain average reference, so this axis is informative but not a straightforward "more sophisticated = better" comparison.

### Key methods references supporting the above (beyond the review's own 65-paper extraction)
- Klug & Gramann (2021), *Eur J Neurosci* 54:8406 — high-pass cutoff × channel count × mobility on ICA decomposition quality.
- Delorme (2023), *Sci Rep* 13:2372, "EEG is better left alone" — benchmark of EEGLAB/FieldTrip/MNE/Brainstorm pipelines; only high-pass filtering and bad-channel interpolation reliably improved outcomes; some referencing/advanced-baseline choices could hurt.
- Chang et al. (2018, 2020), *IEEE TNSRE* — ASR optimal cutoff range (k ≈ 20–30).
- Anders et al. (2020), *Med Biol Eng Comput* — ASR cutoff × motor-task interaction; gait needs a stricter cutoff than quiet standing; benefit plateaus above k≈10.
- Gonsisko et al. (2022) — argues ASR must be tuned per dataset, reinforcing ASR cutoff as a genuine (if under-reported) forking path.
- Gwin et al. (2011), *NeuroImage* — origin of gait time-warping and whole-cycle ERSP baseline conventions.
- Kline & Snyder (2015 corpus papers) — ASR's benefit is speed-dependent, negligible at slow walking speeds (relevant to the ASR on/off justification above).
- Jas et al. (2017), *NeuroImage* — autoreject, the data-driven epoch-rejection alternative.
- de Cheveigne (2020), *NeuroImage* — ZapLine line-noise removal method.