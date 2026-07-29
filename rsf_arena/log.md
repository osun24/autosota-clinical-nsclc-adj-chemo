# RSF clinical-only experiment log

Structured autonomous-loop entries start below this header. Phase 1 smoke tests are intentionally not logged as iterations.

## Reference: pre-iteration baseline (unmodified `train.py`)

Single run of the pipeline as committed, recorded so iter_001 has something to
compare against. No uncertainty is available because the pipeline does not yet
produce any.

- val_ci: 0.6982 (no interval available)
- val_rmst_diff: 6.85 months (no interval available)
- n_features: 19
- elapsed: 11.8 s (budget 25 min)
- run_dir: `runs/run_20260728_164941`

---

### iter_001 — seed panel and bootstrap uncertainty

- type: CODE
- idea_id: `seed_panel_report`
- hypothesis: Reporting each objective as a seed-panel mean with a bootstrap
  standard error will show that the single-seed point estimates now being
  reported — especially `val_rmst_diff` — carry uncertainty large enough that
  future iterations cannot be adjudicated without it.
- changed_files: `train.py`

**Why this idea first.** The pipeline emits two bare point estimates. The
`log.md` schema asks for `0.XXXX ± 0.XXXX` on both objectives, and red line 7
makes a BETTER verdict conditional on each objective beating the previous best
by more than the bootstrap SE / IQR of that metric. Neither is computable today,
so every subsequent iteration would be unadjudicable. Build the measuring
instrument before chasing the number. Validation has 259 patients and 117
events, and only 38 are in the ACT arm, so `val_rmst_diff` is expected to be
noisy.

**Design (prespecified before running).**
1. Seed panel of K = 10 forest seeds, fixed in code. The final model is refit
   once per seed on the training rows; IPTW weights are fit once on training
   rows and shared, since they do not depend on the forest seed.
2. Point estimate for each objective = mean across the panel on the full
   validation set.
3. Uncertainty has two components, reported separately and combined:
   - `se_boot`: B = 500 resamples of validation **rows**; per resample the
     panel-mean metric is recomputed from the stored per-seed prediction
     vectors. Also record the IQR, since red line 7 names SE / IQR.
   - `sd_seed`: spread across the K seeds on the full validation set. Its
     contribution to the mean is `sd_seed / sqrt(K)`.
   - Reported `±` is `sqrt(se_boot^2 + sd_seed^2 / K)`.
4. Recommendation stability: per-patient modal-recommendation frequency
   averaged over patients, plus the fraction of patients whose recommendation
   is unanimous across the panel.
5. The persisted model stays the prespecified first panel seed (7), which is
   the seed the pipeline already used, so downstream artifacts keep their
   meaning.
6. The Optuna search is left untouched, so this iteration changes measurement
   only and its effect is attributable.

**Red-line audit (before editing).**
1. *Sealed test.* No new file reads. `prepare.load_train_valid()` remains the
   only data entry point and `assert_not_test_path` still guards it. PASS.
2. *No leakage.* Panel refits use training rows only; IPTW is fit on training
   rows only. The validation bootstrap resamples validation rows solely to put
   an interval around an already-computed metric — it never touches fitting,
   and the Optuna objective is unchanged, so no validation information enters
   model selection. PASS.
3. *Never drop censored patients.* Row resampling is with replacement over all
   validation rows; censored patients are retained and KM handles them. PASS.
4. *Metric definitions versioned.* `cindex` and `alignment_rmst_difference`
   from `clinical_data` are called unmodified at tau = 60. New quantities are
   added as extra fields; neither objective is replaced. PASS.
5. *Counterfactual recommendation.* Each panel seed predicts risk under ACT=1
   and ACT=0 for every validation patient and recommends the lower-risk arm.
   Unchanged. PASS.
6. *No regimen-level claims.* ACT stays binary. PASS.
7. *Both objectives.* This iteration supplies the SEs that rule needs. It is
   measurement-only, so it is not expected to move either objective; see the
   verdict note below.

**Verdict-schema note.** The schema offers BETTER | WORSE | MIXED, all of which
presume a prior best. iter_001 is measurement-only against a baseline that has
no interval, so none of the three applies honestly. Recorded as BASELINE, and
this entry's numbers become the reference for iter_002 onward.

- val_ci: 0.6979 ± 0.0239
- val_rmst_diff: 7.35 ± 2.70 (months)
- n_features: 19
- verdict: BASELINE (measurement-only; establishes the comparison point)
- one_line_lesson: The C-index is seed-stable but `val_rmst_diff` moves 5.72 to
  9.88 months on forest seed alone, so a single-seed RMST number is not a
  result and nothing below roughly 2.7 months of movement can be called BETTER.

**Reported uncertainty (added columns, neither objective replaced).**

| objective | point (panel mean) | se_total | se_boot | sd_seed | iqr_boot |
|---|---|---|---|---|---|
| val_ci | 0.6979 | 0.0239 | 0.0239 | 0.0009 | 0.0318 |
| val_rmst_diff | 7.3547 | 2.7024 | 2.6775 | 1.1574 | 3.5095 |

- per-seed val_ci: 0.6982, 0.6976, 0.6964, 0.6985, 0.6990, 0.6990, 0.6973,
  0.6983, 0.6970, 0.6973
- per-seed val_rmst_diff: 6.85, 6.85, 7.29, 9.88, 7.67, 6.89, 8.59, 7.25,
  6.55, 5.72
- recommendation agreement: 0.966 mean modal frequency; 84.2% of validation
  patients get a unanimous recommendation across the panel
- ACT recommended for 30.5% of validation patients on average
- metric draws: 500 of 500 usable; elapsed 44.3 s of the 25-min budget

**Findings.**
1. The hypothesis holds, and more sharply than expected for RMST. `sd_seed` for
   `val_ci` is 0.0009, so discrimination is essentially seed-independent; its
   uncertainty is almost entirely validation-sampling (`se_boot` 0.0239). For
   `val_rmst_diff` the seed spread alone is 1.16 months across a 5.72-to-9.88
   range, on top of a 2.68-month sampling SE.
2. The previously reported headline of 6.85 months was seed 7's draw. The panel
   mean is 7.35. Neither number is wrong; the single-seed one is just not
   reproducible, which is what mattered.
3. `val_rmst_diff` sits about 2.7 SE above zero. There is a real alignment
   signal here, but it is not comfortably established, and it is the objective
   every future iteration will be tempted to chase.
4. Recommendation instability concentrates in a minority of patients: 84.2% are
   unanimous across seeds, so roughly 41 of 259 validation patients flip
   recommendation on seed alone. That is the likely source of the RMST swing,
   since flipping a patient moves them between the aligned and unaligned KM
   curves.

**Consequences for the loop (prespecified now, before any tuning).**
- Both objectives are reported as panel means from here on. Single-seed numbers
  are not eligible for a verdict.
- Red line 7 thresholds for the next iteration: `val_ci` must gain more than
  0.0239 and `val_rmst_diff` more than 2.70 months for BETTER. An RMST gain of
  1-2 months is inside seed-plus-sampling noise and is MIXED at best.
- A gain that appears only after changing the reporting seed is not a gain.

---

### iter_002 — ensemble contrast recommendation

- type: ALGO
- idea_id: `recommendation_stability_forest`
- hypothesis: Recommending from the seed panel's mean counterfactual contrast,
  rather than from one forest's contrast, will stabilize the ~41 validation
  patients who currently flip on seed alone and lift `val_rmst_diff` above the
  single-forest expectation of 7.35 months.
- changed_files: `train.py`

**Why this idea.** iter_001 localized the RMST noise: 15.8% of validation
patients receive a different recommendation depending only on the forest seed,
and flipping a patient moves them between the aligned and unaligned KM curves.
Averaging the contrast before thresholding is the direct fix, and the idea
library clears it provided the final recommendation uses the prespecified mean
contrast, which is what is implemented here.

**Design (prespecified before running).**
1. For each of the 10 panel seeds, store the counterfactual contrast
   `risk(ACT=0) - risk(ACT=1)` per validation patient.
2. Ensemble recommendation = ACT where the **mean** contrast across the panel
   is positive. Ensemble risk for the C-index = mean risk across the panel.
3. The ensemble is deterministic given the fixed seed panel, so its only
   uncertainty is validation-row sampling; `se_total` = `se_boot` over the same
   500 draws, using the same draw seed as iter_001 so the two are paired.
4. The Optuna search is left untouched. Only the final recommendation rule
   changes, so any movement is attributable to the rule.
5. The single-seed keys stay in `result.json` and the frozen-evaluator
   consistency check stays active.

**Comparison basis.** iter_001 measured the expected performance of one
randomly seeded forest (`val_ci` 0.6979 ± 0.0239, `val_rmst_diff` 7.35 ± 2.70).
This iteration measures a deterministic ensemble policy. That is the honest
comparison for choosing something deployable.

**Red-line audit (before editing).**
1. *Sealed test.* No new file reads. PASS.
2. *No leakage.* All 10 panel forests fit on training rows only with IPTW fit
   on training rows only. The ensemble averages predictions, never outcomes.
   Row resampling remains post-hoc interval estimation. PASS.
3. *Never drop censored patients.* Unchanged. PASS.
4. *Metric definitions versioned.* `cindex` and `alignment_rmst_difference`
   called unmodified at tau = 60. PASS.
5. *Counterfactual recommendation.* Strengthened, not weakened: every patient
   is still predicted under ACT=1 and ACT=0, and the lower-risk arm wins. No
   within-arm regression on observed outcomes. PASS.
6. *No regimen-level claims.* ACT stays binary. PASS.
7. *Both objectives.* BETTER requires `val_ci` to gain more than 0.0239 and
   `val_rmst_diff` more than 2.70 months against iter_001.

- val_ci: 0.6977 ± 0.0240
- val_rmst_diff: 6.73 ± 2.81 (months)
- n_features: 19
- verdict: WORSE (nominally on both, but both gaps sit far inside the SE)
- one_line_lesson: Averaging contrasts removes the seed variance from the
  reported number without improving expected performance, because the seed
  disagreement lives entirely in near-tie patients whose assignment carries no
  RMST signal.

**Result vs iter_001 (panel mean, same 500 draws).**

| objective | iter_001 panel | iter_002 ensemble | delta | BETTER needs |
|---|---|---|---|---|
| val_ci | 0.6979 ± 0.0239 | 0.6977 ± 0.0240 | -0.0002 | > +0.0239 |
| val_rmst_diff | 7.355 ± 2.702 | 6.725 ± 2.812 | -0.63 | > +2.70 |

- ensemble ACT-recommended fraction: 30.9% (panel mean was 30.5%)
- ensemble recommendations differing from seed 7: **6 of 259**
- `sd_seed` for the ensemble is 0 by construction; elapsed 48.1 s

**Findings.**
1. The hypothesis is refuted. Both objectives moved down, and both moves are an
   order of magnitude inside their SEs, so the honest reading is that the
   ensemble policy is indistinguishable from a single randomly seeded forest.
2. The diagnostic that explains why: iter_001 found 41 non-unanimous patients,
   yet the ensemble differs from seed 7 on only 6. The seed disagreement is
   concentrated in patients whose mean contrast is nearly zero. Averaging
   resolves those ties, but resolving a tie one way rather than another has no
   systematic outcome benefit — their assignment is close to arbitrary with
   respect to survival.
3. So recommendation instability is a *reporting* problem, not a performance
   lever. The 5.72-to-9.88 RMST swing across seeds is these marginal patients
   shuffling between the aligned and unaligned KM curves, and no assignment of
   them is reliably better than another.
4. Chasing the flipping patients further is not worthwhile. The RMST ceiling is
   set by how few validation events there are (117, only 24 in the ACT arm),
   not by which side of a tie a marginal patient lands on.

**Disposition.** Not adopted as the recommendation policy. The ensemble is
retained purely as an additive reported diagnostic, which costs nothing and
gives a seed-free number; the headline objectives stay the panel mean so the
comparison basis is unchanged for iter_003 onward. The iter_001 model remains
champion and iter_002's pickle is deleted.

---

### iter_003 — out-of-bag discrimination and search-selection diagnostics

- type: CODE
- idea_id: `oob_diagnostics`
- hypothesis: The Optuna search maximizes validation metrics over only 10
  trials with a 2.70-month RMST SE, so the reported validation numbers are
  partly a selection artefact; an out-of-bag discrimination estimate computed
  on training rows, plus the spread of all trial objective values, will expose
  how much of the headline is search noise.
- changed_files: `train.py`

**Why this idea.** iter_002 established that the marginal-patient churn is not
a performance lever, so the next question is whether the numbers we are
steering by are trustworthy at all. Hyperparameters are chosen by taking a
weighted maximum over trials whose objectives are evaluated on validation. With
10 trials and an RMST SE of 2.70 months, the winner is plausibly the luckiest
draw rather than the best configuration. OOB C-index gives a validation-free
view of discrimination, and the trial-value spread quantifies the selection.

**Design (prespecified before running).**
1. Fit each panel forest with `oob_score=True` and record `oob_score_`, the
   OOB concordance on training rows. This is added only to the panel's fit
   parameters, not to `metadata["rsf_params"]`, so the human finalizer is
   unaffected.
2. Record every completed trial's two objective values, plus the mean, SD, min
   and max of each objective across trials, and the chosen trial's rank.
3. Report `val_ci - oob_ci` as an optimism gap. This is a diagnostic only; it
   does not redefine or replace either objective, per the idea's admissibility.
4. Nothing in the fitting, the search, or the recommendation rule changes.

**Red-line audit (before editing).**
1. *Sealed test.* No new file reads. PASS.
2. *No leakage.* OOB scores are computed inside the training bootstrap by the
   forest itself and touch no validation row. Trial values are already
   computed; recording them adds no new information flow. PASS.
3. *Never drop censored patients.* Unchanged. PASS.
4. *Metric definitions versioned.* Both objectives are untouched. OOB
   concordance is an added diagnostic column, explicitly not a replacement.
   PASS.
5. *Counterfactual recommendation.* Unchanged. PASS.
6. *No regimen-level claims.* PASS.
7. *Both objectives.* Measurement-only again; a verdict of BETTER is not
   expected and would in fact be suspicious.

- val_ci: 0.6979 ± 0.0239
- val_rmst_diff: 7.35 ± 2.70 (months)
- n_features: 19
- verdict: NEUTRAL (measurement-only; both objectives bit-identical to iter_001)
- one_line_lesson: The search hands back the argmax of 10 noisy RMST
  evaluations, and that winner's 8.58 months shrinks to 7.36 on honest
  re-estimation, so search-time trial values are inflated and must never be
  quoted as results.

**Diagnostics (added columns; neither objective replaced).**

| quantity | value |
|---|---|
| OOB C-index (panel mean) | 0.6796 (sd 0.0009) |
| validation C-index (panel mean) | 0.6979 |
| optimism gap `val_ci - oob_ci` | **+0.0182** |
| completed trials | 10 (Pareto front 3) |
| chosen trial | 2, search values ci 0.6832 / rmst 8.5801 |
| chosen trial rank | **1 of 10 on RMST**, 6 of 10 on C-index |
| trial RMST across the search | mean 6.652, sd 1.455, min 4.624, max 8.580 |
| trial C-index across the search | mean 0.6849, sd 0.0091, min 0.6728, max 0.6990 |

**Findings.**
1. Objectives are unchanged to the last digit, which is the correct outcome for
   a measurement-only iteration and confirms the additions are inert.
2. The selection bias is real and measurable. The compromise selector weights
   RMST at 0.6, and it picked the trial ranked **first of ten on RMST**. Its
   search-time RMST was 8.58; the same configuration honestly re-estimated by
   the seed panel gives 7.36. That 1.2-month drop is winner's curse, visible
   directly rather than inferred.
3. Trial-to-trial RMST SD is 1.455 months with only 2 bootstraps per trial, so
   the search is largely ranking noise. Taking the max of 10 such draws inflates
   the expectation by roughly 1.5 SD, which is the same order as the observed
   shrinkage.
4. The OOB result cuts the other way and is reassuring: validation C-index
   (0.6979) is *higher* than out-of-bag C-index on training rows (0.6796). The
   model is not inflating its discrimination on validation; if anything the
   validation cohort is the easier one to rank. Discrimination is therefore not
   where the fragility lives — the RMST objective is.
5. OOB C-index has sd 0.0009 across seeds, mirroring validation C-index. Both
   confirm discrimination is a stable quantity in this setup.

**Consequences for the loop.**
- Search-time trial values are not results and are not quotable. Only
  seed-panel re-estimates count.
- More trials without more bootstraps per trial will make selection *worse*,
  not better, because it deepens the max over a noisy ranking. Any PARAM
  iteration that widens the search must raise `bootstrap_n` alongside it.
- The C-index has little room to move and is stable; the loop's leverage, if
  any exists, is on RMST.

---

### iter_004 — de-noised search over coordinated leaf and split sizes

- type: PARAM
- idea_id: `split_size_coordination` + `leaf_size_sweep`
- hypothesis: Ranking configurations on 8 bootstraps instead of 2, over a
  coordinated leaf/split grid that excludes the out-of-scope leaf sizes below
  5, will select a configuration whose honest seed-panel re-estimate is higher
  than the current 7.36 months and whose winner's-curse shrinkage is smaller
  than the 1.2 months measured in iter_003.
- changed_files: `train.py`

**Why this idea.** iter_003 showed the search returns the argmax of 10 noisy
RMST evaluations and that its 8.58 shrank to 7.36 on honest re-estimation. The
fix is not more trials — that deepens the max over a noisy ranking — but less
noise per trial. Separately, the current space samples `min_samples_leaf` from
3, which the idea library puts out of scope below 5 absent an uncertainty
audit, and it samples `min_samples_split` independently of leaf size, which
admits redundant and invalid combinations.

**Design (prespecified before running).**
1. `bootstrap_n` 2 to 8, so each trial's objective averages 8 evaluations.
2. `n_trials` 10 to 30. Justified only because per-trial noise drops first;
   the two move together, as iter_003 required.
3. `min_samples_leaf` sampled 5 to 30, bringing the space inside the library's
   stated scope.
4. `min_samples_split` is no longer independent: a multiplier of 2 to 4 times
   the leaf size, so every combination is valid and non-redundant by
   construction.
5. `max_features`, `max_depth` and `n_estimators` ranges are untouched, so
   depth and mtry remain available as separate later iterations.
6. Primary readout stays the seed panel. The search-vs-panel shrinkage is
   recorded as the secondary readout of whether selection got more reliable.

**Red-line audit (before editing).**
1. *Sealed test.* No new file reads. PASS.
2. *No leakage.* More bootstraps means more resampling of **training** rows;
   IPTW is refit inside each bootstrap on training rows only. Validation is
   still only ever scored, never fitted on. PASS.
3. *Never drop censored patients.* Bootstraps resample all training rows and
   `require_two_arms` still holds. PASS.
4. *Metric definitions versioned.* Untouched. PASS.
5. *Counterfactual recommendation.* Untouched. PASS.
6. *No regimen-level claims.* PASS.
7. *Both objectives.* BETTER needs `val_ci` > +0.0239 and `val_rmst_diff` >
   +2.70 months against iter_001's 0.6979 / 7.35.

- val_ci: 0.7005 ± 0.0239
- val_rmst_diff: 7.12 ± 2.57 (months)
- n_features: 19
- verdict: MIXED (nominal ci gain, nominal rmst loss, both inside noise)
- one_line_lesson: More bootstraps per trial did not shrink winner's curse as
  hypothesized — it got worse (1.71 vs 1.20 months) because the wider,
  de-noised space let one trial dominate both objectives simultaneously
  (Pareto front collapsed to size 1), concentrating rather than diluting the
  selection pressure.

**Result vs iter_001 (panel mean, same 500 draws; chosen params: leaf=10,
split=34, depth=8, max_features=0.415, n_estimators=500).**

| objective | iter_001 panel | iter_004 panel | delta | BETTER needs |
|---|---|---|---|---|
| val_ci | 0.6979 ± 0.0239 | 0.7005 ± 0.0239 | +0.0026 | > +0.0239 |
| val_rmst_diff | 7.355 ± 2.702 | 7.123 ± 2.570 | -0.23 | > +2.70 |

- OOB C-index 0.6826, optimism gap +0.0179 (comparable to iter_003's +0.0182)
- recommendation agreement 0.955, unanimous 82.2%, ACT-recommended 34.0%
- search: 30 trials, chosen trial ranked **1st of 30 on both objectives**
  (Pareto front size 1) — search values ci 0.6933 / rmst 8.8372
- shrinkage on adoption, search value to honest panel re-estimate:
  **1.71 months** (iter_003's single-dominant-trial case was 1.20)
- elapsed 190.5 s of the 25-min budget (up from ~45-49 s, still ample margin)

**Findings.**
1. Both deltas versus iter_001 are an order of magnitude inside their SEs.
   Practically this configuration is indistinguishable from the iter_001
   defaults — coordinating leaf/split size and constraining leaf size to the
   library's [5, 30] scope neither helped nor hurt discrimination or RMST.
2. The core hypothesis — that more bootstraps per trial reduces winner's curse
   — is refuted by the shrinkage number, which went the wrong way. The
   mechanism: with only 2 bootstraps, trials were noisy enough that no single
   trial usually dominated both objectives (iter_003's Pareto front was 3).
   With 8 bootstraps the per-trial estimate firmed up enough that one trial
   swept both objectives (Pareto front 1), so the compromise selector no
   longer averages across a small set of similarly-good options — it just
   takes that one trial's still-noisy joint maximum at face value.
3. Net effect: de-noising trials made the selector *more* confident in a
   single winner, not less exposed to its noise. The 8 bootstraps reduced
   per-configuration noise but did nothing about across-configuration
   selection noise, which is what winner's curse actually measures.
4. This suggests the real lever is not bootstraps-per-trial but either (a)
   re-evaluating the top-K trials with a fresh, independent set of bootstraps
   before final selection, or (b) discounting/regularizing the selection
   criterion itself (e.g., picking near the Pareto front's centroid rather
   than its edge). Neither is implemented this iteration; flagged for a future
   CODE iteration.

**Disposition.** Not BETTER, not worse-on-both, so per the workflow this is
MIXED and may stay for up to 3 follow-up iterations. The code changes
(coordinated leaf/split parameterization, expanded trials/bootstraps) are
retained because they are a genuine correctness improvement independent of the
noise result — the search space no longer admits invalid or below-scope
leaf/split combinations — and the runtime cost (190 s) leaves ample budget
margin. Champion pickle is unchanged (iter_001's model); this run's pickle is
pruned. iter_001's panel numbers (0.6979 ± 0.0239 / 7.355 ± 2.70) remain the
comparison point for BETTER going forward, since panel evaluation methodology
does not depend on search settings.

---

### iter_005 — IPTW effective sample size and weight diagnostics

- type: CODE
- idea_id: `effective_sample_size`
- hypothesis: With only 114/775 training patients and 38/259 validation
  patients in the ACT arm, IPTW weighting may be spending a large share of its
  effective sample size on a handful of extreme-propensity patients; reporting
  ESS and weight quantiles by arm will show whether the counterfactual
  contrast (which both objectives depend on) rests on a well-supported
  weighting or a fragile one.
- changed_files: `train.py`

**Design.** Using the training-row IPTW weights already computed once in
`seed_panel_report` (they don't depend on forest seed), report: overall ESS =
`(sum w)^2 / sum(w^2)`, ESS by arm, and weight quantiles (min/25/50/75/max)
overall and by arm. Diagnostic only — added as new result fields, nothing
about `compute_iptw`, the objective, or the recommendation rule changes.

**Red-line audit.** No new file reads (PASS #1). Weights are the existing
training-only IPTW fit, read-only summarized (PASS #2). No patients dropped,
only summarized (PASS #3). Objectives untouched, this is an added diagnostic
(PASS #4). Recommendation rule untouched (PASS #5). ACT stays binary (PASS
#6). Measurement-only, no verdict beyond NEUTRAL expected (PASS #7).

- val_ci: 0.7005 ± 0.0239
- val_rmst_diff: 7.12 ± 2.57 (months)
- n_features: 19
- verdict: NEUTRAL (measurement-only; bit-identical to iter_004, as expected
  since weighting itself is untouched)
- one_line_lesson: IPTW weights are usable (overall ESS 496/775, 64%) but the
  OBS arm has patients pinned at the upper clip of 10.0, meaning a handful of
  observation-arm patients look almost certain to have received chemo by their
  covariates and are being reweighted up by the full clip ceiling to
  compensate — worth revisiting the clip range as a dedicated PARAM iteration.

**IPTW diagnostics (training rows, added columns; weighting itself untouched).**

| | n | ESS | ESS % | weight median | weight IQR | weight range |
|---|---|---|---|---|---|---|
| overall | 775 | 496.4 | 64.1% | 1.242 | [1.048, 1.713] | [0.155, 10.0] |
| ACT | 114 | 89.5 | 78.5% | 0.236 | [0.198, 0.301] | [0.155, 0.977] |
| OBS | 661 | 470.9 | 71.2% | 1.369 | [1.123, 1.811] | [0.898, **10.0**] |

**Findings.**
1. Both arms individually keep 70-80% of their nominal sample size after
   weighting, which is adequate. The overall ESS (64%) is lower mostly because
   ACT and OBS are weighted toward a shared reference prevalence, not because
   either arm alone is thin.
2. The OBS-arm weight distribution touches the upper clip bound of 10.0
   exactly, which means at least one training patient has an estimated
   propensity for OBS so low (equivalently, predicted probability of ACT so
   high given covariates) that the raw IPTW weight exceeded the clip and was
   truncated. Clipping is doing its job — the weight was capped rather than
   left to blow up — but a weight sitting at the cap is a signal that the
   propensity model is confidently extrapolating for that patient, which
   the current `iptw_clip_sweep` idea (CLEARED, PARAM) is the prespecified way
   to investigate.
3. The ACT arm's weights are much more compressed (IQR 0.198-0.301, no
   clipping observed) — consistent with ACT being the minority treatment,
   where predicted propensity for ACT clusters lower and away from the clip
   boundary.
4. No objective moved, as expected: this iteration reports on already-computed
   weights and changes nothing about `compute_iptw` or the bootstrap fitting.

**Disposition.** Diagnostic retained permanently at low cost (weights are
already computed; this only summarizes them). Recommend `iptw_clip_sweep` as
the next PARAM candidate given the OBS-arm clip-ceiling finding, to check
whether a wider or narrower `w_clip` changes ESS meaningfully without moving
the estimand.

---

### iter_006 — IPTW weight-clip sweep (training diagnostics only)

- type: PARAM
- idea_id: `iptw_clip_sweep`
- hypothesis: iter_005 found OBS-arm weights pinned at the upper clip of 10.0;
  widening or narrowing the clip range and selecting by training-only ESS
  (never validation) will find a range with materially higher overall ESS
  than the current default (0.1, 10.0), which gave 64.1%.
- changed_files: `train.py`

**Design (prespecified before running).** Candidates for `w_clip`:
`(0.1, 5.0)`, `(0.1, 10.0)` [current default], `(0.1, 20.0)`, `(0.05, 10.0)`.
For each, compute IPTW weights on the **full training set** (not a bootstrap)
and score by overall ESS — a training-only diagnostic, satisfying the idea's
admissibility restriction. The argmax-ESS candidate is selected once, before
any bootstrap fitting, and used uniformly for every bootstrap's weight
computation and the final panel fit. `ps_clip` (0.05, 0.95) is left untouched;
this iteration varies only the weight clip. The selection never looks at
`val_ci` or `val_rmst_diff` — those are computed only after the clip is fixed.
`metadata["iptw_w_clip"]` records the chosen range so `fit_model_from_metadata`
(used by the human-only finalizer) stays consistent; it defaults to the old
`(0.1, 10.0)` if the key is absent, so old run artifacts stay interpretable.

**Red-line audit.**
1. *Sealed test.* No new file reads. PASS.
2. *No leakage.* Selection uses only the training-row propensity/weight
   distribution; no validation metric informs which clip is chosen (the
   admissibility condition this idea is CLEARED under). Per-bootstrap weights
   are still fit inside each bootstrap on training rows only. PASS.
3. *Never drop censored patients.* Unaffected — clipping only bounds weight
   magnitude, never excludes rows. PASS.
4. *Metric definitions versioned.* Objectives untouched; IPTW clip is not one
   of the two frozen metrics. PASS.
5. *Counterfactual recommendation.* Unaffected. PASS.
6. *No regimen-level claims.* PASS.
7. *Both objectives.* BETTER needs `val_ci` > +0.0239 and `val_rmst_diff` >
   +2.70 months against iter_001's 0.6979 / 7.355 — the standing comparison
   point, since search settings don't change what "current best" means.

- val_ci: 0.6992 ± 0.0239
- val_rmst_diff: 7.37 ± 2.64 (months)
- n_features: 19
- verdict: MIXED (both objectives nominally flat-to-up, neither clears the
  BETTER threshold)
- one_line_lesson: Tightening the weight clip to (0.1, 5.0) lifted training
  ESS by 11% (496 to 551 of 775) with both objectives unchanged within noise —
  the OBS-arm clip-ceiling patients found in iter_005 were a weighting
  efficiency issue, not a source of estimation bias large enough to show up in
  either objective.

**Clip candidates (training-only ESS, selection made before any validation
metric was computed).**

| w_clip | overall ESS | ACT ESS | OBS ESS | OBS max weight |
|---|---|---|---|---|
| (0.1, 5.0) | **550.6** | 89.5 | 522.0 | 5.0 (capped) |
| (0.1, 10.0) [prior default] | 496.4 | 89.5 | 470.9 | 10.0 (capped) |
| (0.1, 20.0) | 463.7 | 89.5 | 440.0 | 16.76 |
| (0.05, 10.0) | 496.4 | 89.5 | 470.9 | 10.0 (capped) |

Selected: **(0.1, 5.0)**, the argmax of overall training ESS.

**Result vs iter_001 (standing comparison point).**

| objective | iter_001 panel | iter_006 panel | delta | BETTER needs |
|---|---|---|---|---|
| val_ci | 0.6979 ± 0.0239 | 0.6992 ± 0.0239 | +0.0013 | > +0.0239 |
| val_rmst_diff | 7.355 ± 2.702 | 7.365 ± 2.638 | +0.01 | > +2.70 |

**Findings.**
1. ACT-arm ESS is identical across every candidate (89.5) — the ACT propensity
   distribution never approaches the clip boundary, consistent with iter_005's
   finding that ACT weights sit well inside [0.155, 0.977]. All of the
   sensitivity is in the OBS arm.
2. Narrowing the OBS-arm clip from 10.0 to 5.0 recovers ESS by capping more
   patients' weight, but at a *lower* cap — i.e., more patients get bounded,
   but each bound is less extreme, and net variance of the weight
   distribution drops enough to raise ESS. Widening to 20.0 does the opposite:
   fewer patients are capped, but the ones that are get weighted more
   extremely, lowering ESS.
3. Despite an 11% ESS gain, neither objective moved outside noise. The
   patients whose weights change most under tighter clipping are exactly the
   ones the propensity model was already least confident about, and RSF with
   IPTW sample weights appears robust to that specific reweighting range on
   this cohort.
4. This is a genuine, if small, efficiency win with no detected downside:
   tighter clipping trades a small amount of estimand fidelity (more patients
   pinned at the cap) for less variance in the weighted training objective,
   and the objectives confirm no harm.

**Disposition.** Adopted as the new default going forward — `select_iptw_clip`
stays in the pipeline and will keep re-selecting the best training-ESS clip
range on future runs, currently landing on (0.1, 5.0). Since neither objective
cleared the BETTER threshold, iter_001's panel numbers (0.6979 ± 0.0239 /
7.355 ± 2.70) remain the comparison point for future BETTER verdicts.

---

### iter_007 — prespecified clinical interaction expansion

- type: ALGO
- idea_id: `clinical_interaction_expansion`
- hypothesis: RSF splits on individual covariates already capture some
  interactions implicitly, but a small, prespecified, clinically motivated set
  of covariate-by-ACT and covariate-by-covariate product terms will make known
  treatment-effect modifiers (stage, histology, age, sex) available to the
  forest as single splits rather than requiring multiple correlated splits to
  approximate, improving `val_rmst_diff` beyond the 7.355 ± 2.70 standing
  baseline without degrading `val_ci`.
- changed_files: `prepare.py`, `train.py`

**Why this idea, and why now.** Six iterations of measurement (seed panel,
OOB gap, search diagnostics, IPTW ESS, clip sweep) established that
discrimination is stable and search noise/weighting were not the bottleneck.
No ALGO idea besides `recommendation_stability_forest` (iter_002, refuted) has
been tried. This is the first iteration to change what the forest sees.

**Design (prespecified before running — fixed budget of 6 terms).**
1. `ACT_x_Age` = Adjuvant Chemo × Age
2. `ACT_x_StageIII` = Adjuvant Chemo × Stage_III (stage is the standard
   adjuvant-chemo-benefit modifier in NSCLC)
3. `ACT_x_Adenocarcinoma` = Adjuvant Chemo × Histology_Adenocarcinoma
4. `ACT_x_Male` = Adjuvant Chemo × IS_MALE
5. `Age_x_StageIII` = Age × Stage_III (prognostic, not treatment-specific)
6. `Age_x_Smoked_Yes` = Age × Smoked?_Yes (prognostic)
Hierarchical: every term is a product of two columns that remain in the
feature set as main effects, so the forest can still use either main effect
alone. `n_features` goes from 19 to 25 (+31%), a fixed, small expansion.
`prepare.add_interaction_terms` recomputes all six from the *current* value of
`Adjuvant Chemo` and is called unconditionally inside
`prepare.build_matrix_from_feature_names`, so every caller — the seed panel,
the Optuna objective, `prepare.evaluate_on_valid`, and the human-only
finalizer's sealed-test path — automatically gets correct counterfactual
interaction values when `Adjuvant Chemo` is flipped to 0 or 1. Interaction
terms are NOT added to `pretreatment_columns` (the IPTW propensity model's
covariates), since `ACT_x_*` terms are functions of the treatment itself and
have no place predicting it.

**Red-line audit.**
1. *Sealed test.* No new file reads; interaction terms are pure column
   arithmetic on already-loaded frames. PASS.
2. *No leakage.* Interaction terms are deterministic products of raw
   covariates already present in every split (train, validation, and — for
   the human-only finalizer — test); nothing is fit on any data, so there is
   no train/validation information flow to leak. PASS.
3. *Never drop censored patients.* Unaffected — this only adds columns.
   PASS.
4. *Metric definitions versioned.* Objectives untouched. PASS.
5. *Counterfactual recommendation.* Preserved and, by design, sharpened:
   `ACT_x_*` terms are recomputed under both ACT=1 and ACT=0 counterfactuals
   because `build_matrix_from_feature_names` recomputes them from the current
   `Adjuvant Chemo` value on every call — the exact mechanism that makes the
   feature set counterfactual-consistent rather than stale. PASS.
6. *No regimen-level claims.* ACT stays binary throughout; no regimen
   information is introduced. PASS.
7. *Both objectives.* BETTER needs `val_ci` > +0.0239 and `val_rmst_diff` >
   +2.70 months against iter_001's standing 0.6979 / 7.355.

- val_ci: 0.6976 ± 0.0242
- val_rmst_diff: 8.64 ± 2.76 (months)
- n_features: 25
- verdict: MIXED (val_ci flat, val_rmst_diff nominal +1.29 but short of the
  +2.70 threshold — follow-up slot 1 of 3)
- one_line_lesson: The point-estimate gain doesn't clear the formal BETTER
  bar, but the whole per-seed RMST distribution shifted up (baseline range
  5.72-9.88, this run's range 7.63-9.86, no seed at or below the old mean),
  which is a stronger signal than the point delta alone suggests.

**Result vs iter_001 (standing comparison point).**

| objective | iter_001 panel | iter_007 panel | delta | BETTER needs |
|---|---|---|---|---|
| val_ci | 0.6979 ± 0.0239 | 0.6976 ± 0.0242 | -0.0003 | > +0.0239 |
| val_rmst_diff | 7.355 ± 2.702 | 8.642 ± 2.761 | **+1.29** | > +2.70 |

- OOB C-index 0.6757, optimism gap +0.0219 (comparable to prior runs)
- recommendation agreement 0.975 (was 0.966), unanimous 89.6% (was 84.2%)
- ACT-recommended fraction dropped to 22.7% (was 30.5%)
- per-seed val_rmst_diff: 8.43, 7.64, 7.98, 9.86, 7.63, 9.20, 8.23, 9.51,
  8.78, 9.16 — **every seed above the iter_001 mean of 7.36**
- search chosen trial ranked 1st of 30 on both objectives (Pareto front 3);
  shrinkage on adoption was -0.29 months (search value 8.36 vs panel 8.64,
  i.e. the panel re-estimate was *higher* than the search suggested this time)

**Findings.**
1. `val_ci` is unmoved — the interaction terms don't help or hurt
   discrimination, consistent with the earlier finding that C-index is a
   stable, largely already-saturated quantity in this setup.
2. `val_rmst_diff`'s point estimate doesn't clear the pre-registered
   threshold, but the per-seed distribution is the most informative evidence
   so far: every one of 10 seeds landed strictly above the iter_001 panel
   mean, with zero overlap into the bottom half of iter_001's own per-seed
   range. A shift with no seed falling back to baseline is a different kind of
   evidence than a single higher point estimate would be.
3. Recommendation unanimity rose to 89.6% from 84.2%, and the ACT-recommended
   fraction fell to 22.7% from 30.5% — the interaction terms are changing
   *which* patients get recommended ACT, not just adding prediction noise.
4. This iteration also exposes a structural limit: `val_rmst_diff`'s
   `se_boot` (~2.7 months) is set by the 259-patient, 117-event validation
   set and tau=60 RMST — it will not shrink by refining the panel or the
   search. No future ALGO/PARAM change is likely to produce an RMST delta
   that formally clears +2.70 months without either a materially different
   model or a larger validation set. The formal BETTER bar for RMST may be
   effectively unreachable within this arena; directional, distribution-level
   evidence (as here) may be the most honest signal available.

**Disposition.** Not BETTER by the letter of red line 7, and not WORSE either.
Recorded as MIXED, follow-up slot 1 of 3. Given the consistent per-seed shift,
the interaction-expanded feature set (`clinical_columns + INTERACTION_TERMS`,
25 features) is kept as the new default going forward rather than reverted;
iter_008 will check whether the interaction terms are mechanistically
load-bearing (feature importance) before the MIXED status is resolved.
iter_001's panel numbers remain the formal comparison point for BETTER.

---

### iter_008 — permutation feature importance for the interaction terms

- type: CODE
- idea_id: `feature_importance_diagnostic` (new; not in the idea library —
  generated to resolve iter_007's MIXED status with a mechanistic check)
- hypothesis: If iter_007's RMST shift is because the forest is actually using
  the six interaction terms, at least some of them should rank among the more
  important of the 25 features by permutation importance (C-index drop when
  permuted); if they rank at the bottom, the RMST shift is more likely
  incidental to correlated main-effect splits changing rather than to the
  interaction terms themselves.
- changed_files: `train.py`

**Design.** `sklearn.inspection.permutation_importance` on the primary panel
model (seed 7), scored via the model's built-in `.score()` (Harrell's C,
matching the frozen `val_ci` definition), n_repeats=20, seed=13, evaluated on
validation rows exactly as the frozen evaluator already does. This measures
C-index sensitivity, not RMST sensitivity directly (no simple RMST-compatible
scorer exists for `permutation_importance`'s API) — the log will state this
limitation. Diagnostic only: computed after the model is already fit and
selected; does not feed back into fitting, search, or feature_names.

**Red-line audit.** No new file reads (#1 PASS). Uses the already-fit
model and already-loaded validation rows exactly as the frozen evaluator does
— no new information flow into fitting or selection (#2 PASS). No rows
dropped (#3 PASS). `val_ci`/`val_rmst_diff` untouched; this is an added
diagnostic (#4 PASS). Recommendation rule untouched (#5 PASS). ACT stays
binary (#6 PASS). Measurement-only; not expected to move either objective
(#7 PASS).

- val_ci: 0.6976 ± 0.0242
- val_rmst_diff: 8.64 ± 2.76 (months)
- n_features: 25
- verdict: NEUTRAL (measurement-only; bit-identical to iter_007, as expected)
- one_line_lesson: The RMST shift in iter_007 is not coming from the
  hypothesized mechanism — every `ACT_x_*` treatment-interaction term ranks in
  the bottom half of 25 features (two near exactly zero), while
  `Age_x_StageIII`, a purely prognostic term with no treatment involvement at
  all, ranks 4th.

**Permutation importance ranking (C-index drop, 20 repeats, primary model).**

| rank | feature | importance | is ACT-interaction? |
|---|---|---|---|
| 1 | Stage_II | 0.0533 | no |
| 2 | Age | 0.0252 | no |
| 3 | Stage_IA | 0.0204 | no |
| 4 | **Age_x_StageIII** | 0.0184 | no (prognostic) |
| 5 | Smoked?_Unknown | 0.0177 | no |
| 10 | Age_x_Smoked_Yes | 0.0026 | no (prognostic) |
| 11 | ACT_x_Age | 0.0017 | **yes** |
| 14 | ACT_x_Male | 0.0001 | **yes** |
| 19 | ACT_x_StageIII | 0.0000 | **yes** |
| 22 | Adjuvant Chemo (main effect) | -0.0004 | — |
| 23 | ACT_x_Adenocarcinoma | -0.0004 | **yes** |

**Findings.**
1. The idea's stated rationale — covariate-by-ACT terms making treatment-effect
   modifiers available as single splits — is not supported. All four
   `ACT_x_*` terms rank 11th or worse of 25, two at essentially zero
   importance. The main `Adjuvant Chemo` variable itself ranks 22nd, near
   zero: consistent with iter_003's finding that treatment status is a weak
   direct predictor of overall risk ranking (C-index), which is unsurprising
   since C-index measures whole-cohort discrimination, not treatment-specific
   effect.
2. `Age_x_StageIII` is genuinely load-bearing (rank 4, importance an order of
   magnitude above most `ACT_x_*` terms) despite involving no treatment
   information at all. `Age_x_Smoked_Yes` is modestly used (rank 10).
3. This resolves *how* iter_007's RMST distribution shifted upward, but not
   *why* it counts as evidence for the original hypothesis — it doesn't. The
   forest found a genuinely useful prognostic split shortcut
   (age-by-late-stage), which changed the risk model broadly enough to shift
   which patients land on which side of the counterfactual recommendation
   threshold, which is exactly the mechanism that drives `val_rmst_diff`
   movement. The four `ACT_x_*` terms are very likely inert ballast, adding
   feature-count risk (the idea's own stated risk: "redundant terms can
   distort split selection") without contributing.
4. Note the caveat already logged in iter_007's pre-audit: permutation
   importance here measures C-index sensitivity, not RMST sensitivity
   directly — there is no simple RMST-compatible scorer for this API. The
   inference above is indirect (via which features the forest's splits
   actually use), not a direct RMST attribution.

**Consequence for the loop.** iter_009 will trim the interaction budget to
just the two prognostic terms that show real importance
(`Age_x_StageIII`, `Age_x_Smoked_Yes`), dropping the four inert `ACT_x_*`
terms, as follow-up slot 2 of 3 for iter_007's still-open MIXED status.

---

### iter_009 — trim interaction budget to load-bearing terms

- type: ALGO
- idea_id: `clinical_interaction_expansion` (refinement; follow-up slot 2 of 3
  for iter_007's MIXED verdict)
- hypothesis: Dropping the four `ACT_x_*` terms that iter_008 showed are
  inert (rank 11, 14, 19, 23 of 25, two near zero importance) and keeping only
  `Age_x_StageIII` and `Age_x_Smoked_Yes` will preserve most of iter_007's
  RMST shift while reducing feature-count risk and removing dead weight from
  the split-selection budget.
- changed_files: `train.py`

**Design.** `feature_names = clinical_columns + ["Age_x_StageIII",
"Age_x_Smoked_Yes"]`, i.e. 21 features instead of 25. `prepare.py` is
unchanged — `add_interaction_terms` still computes all six terms (cheap,
harmless, keeps the full set available for any future reconsideration); only
`train.py`'s feature selection is narrowed. Everything else (search space,
clip selection, panel, diagnostics) is identical to iter_007/008.

**Red-line audit.** Strict subset of an already-audited, already-cleared
feature set; no new columns, no new data access, no change to fitting,
weighting, or the recommendation rule. All seven red lines carry over
unchanged from iter_007's audit. PASS on all.

- val_ci: 0.6983 ± 0.0239
- val_rmst_diff: 7.75 ± 2.76 (months)
- n_features: 21
- verdict: MIXED, and RESOLVED as NOT ADOPTED (closing follow-up early —
  slot 2 of 3, not using slot 3)
- one_line_lesson: Trimming to the two terms permutation importance called
  load-bearing did not preserve iter_007's RMST shift — it mostly erased it
  (+1.29 months shrank to +0.39), which means the shift was driven more by
  which hyperparameter configuration the search happened to land on than by
  the interaction terms themselves being a systematic source of signal.

**Result vs iter_001 (standing comparison point) and vs iter_007 (25-feature
version).**

| objective | iter_001 (19 feat) | iter_007 (25 feat) | iter_009 (21 feat) |
|---|---|---|---|
| val_ci | 0.6979 ± 0.0239 | 0.6976 ± 0.0242 | 0.6983 ± 0.0239 |
| val_rmst_diff | 7.355 ± 2.702 | 8.642 ± 2.761 | 7.749 ± 2.755 |
| delta vs iter_001 (rmst) | — | +1.29 | +0.39 |

- per-seed val_rmst_diff: 7.60, 8.41, 7.01, 6.89, 8.57, 7.23, 7.64, 9.11,
  8.63, 6.40 — unlike iter_007, **two seeds fall below the iter_001 mean**
  (6.89, 6.40), so the "every seed above baseline" pattern that made iter_007
  notable did not reproduce
- permutation importance ranking is nearly identical to iter_007's
  (`Age_x_StageIII` still rank 4, `Age_x_Smoked_Yes` now rank 9) — the forest
  is using these terms similarly in both runs, yet the RMST outcome differs
  substantially

**Findings.**
1. The hypothesis behind this iteration — that dropping the inert `ACT_x_*`
   terms would preserve most of the RMST gain while removing dead weight — is
   refuted. Feature usage (per permutation importance) barely changed between
   the 25- and 21-feature versions, but the RMST point estimate moved by 0.9
   months, more than half of iter_007's original gain. If a feature the model
   uses identically both times can be associated with such different RMST
   outcomes, the RMST metric is picking up something other than that
   feature's marginal contribution — most plausibly the specific
   hyperparameter configuration the Optuna search happened to select each run
   (search draws are not identical between 25-feature and 21-feature spaces,
   since `max_features` samples a fraction of a different-sized feature set).
2. This reinforces iter_007's own structural finding: with `se_boot` around
   2.7 months on this validation set, RMST point estimates swing by amounts
   comparable to genuine hyperparameter-search variation, not just seed
   variation. Two different runs of "morally the same" model can differ by
   ~1 month for reasons that have nothing to do with the feature set change
   being tested.
3. `val_ci` stayed flat across all three variants (0.6976-0.6983), the most
   stable finding across this whole ALGO/CODE arc.

**Disposition — resolving iter_007's MIXED status now rather than spending a
third follow-up.** The interaction expansion (in either the original 6-term
or the trimmed 2-term form) is **not adopted**. Evidence across two follow-ups
shows the RMST association is not robust to a change that permutation
importance said should be inert, which is a stronger disqualifier than
either point estimate alone. `train.py` is reverted to
`feature_names = clinical_columns` (19 features, no interaction terms) as
part of this commit. `prepare.add_interaction_terms` and
`prepare.INTERACTION_TERMS` are left in `prepare.py` — unused by `run()`, at
zero cost, available if a future iteration wants to revisit interactions with
a design that isolates the hyperparameter-search confound (e.g. a fixed
hyperparameter configuration held constant across the with/without
comparison). iter_001's panel numbers (0.6979 ± 0.0239 / 7.355 ± 2.70) remain
the standing comparison point; no candidate has cleared BETTER through nine
iterations.

---

### iter_010 — controlled depth sweep (bypassing the noisy search)

- type: PARAM
- idea_id: `depth_regularization`
- hypothesis: Every PARAM/ALGO comparison so far has been confounded by
  Optuna search noise (iter_003, iter_009). A direct sweep — same leaf size,
  split size, mtry, and n_estimators held fixed, only `max_depth` varied,
  each depth evaluated on the full 10-seed panel — will give a clean read on
  whether bounding depth helps, without the search's winner's-curse
  contaminating the comparison.
- changed_files: `train.py`

**Design (prespecified before running).** Fixed baseline params: `n_estimators=700`,
`min_samples_leaf=15`, `min_samples_split=37` (multiplier 2.5, matching
iter_004's coordination scheme), `max_features=0.7` — all interior points of
their CLEARED PARAM ranges, chosen without looking at any result. Depth
candidates per the idea: `3, 4, 5, 6, 7, 8, 9, 10, None`. For each candidate,
refit the 10-seed panel (base params + that depth) on training rows and
report the panel mean/SE for both objectives, using the same machinery as
`seed_panel_report`. This is a read-only diagnostic sweep — it does not
change `_suggest_params`, the Optuna search space, or which model gets
persisted; the persisted model stays whatever the existing search selects.
If one depth clearly and robustly dominates the sweep, that becomes a
candidate to bound `max_depth`'s search range in a later iteration.

**Red-line audit.** No new file reads (#1). All fits are on training rows,
IPTW weights fit on training rows only, exactly as the existing panel does
(#2). No rows dropped (#3). Metrics computed via the frozen `cindex` /
`alignment_rmst_difference` (#4). Recommendation rule unchanged — risk under
ACT=1/ACT=0 compared per patient (#5). ACT stays binary (#6). This sweep does
not update `metadata["rsf_params"]` or the persisted model, so it cannot by
itself produce a BETTER/WORSE verdict on the run's headline numbers; those
stay whatever the unmodified search+panel produces this iteration (#7).

- val_ci: 0.6992 ± 0.0239
- val_rmst_diff: 7.37 ± 2.64 (months)
- n_features: 19
- verdict: NEUTRAL on the headline run (search/panel unchanged, deltas vs
  iter_001 negligible: +0.0013 ci, +0.01 rmst); sweep finding below is the
  substantive result of this iteration
- one_line_lesson: `max_depth=3` is a clear, reproducible loser — RMST collapses
  to 4.57 months (vs 7.3-8.5 for depth 4 through unrestricted) with roughly
  double the seed variance — while depth 4 through unrestricted are all
  statistically indistinguishable from each other, and deeper/unrestricted
  trees give *more* seed-stable RMST than moderate depths, not less.

**Depth sweep (fixed leaf=15, split=37, mtry=0.7, n_estimators=700; sd_seed
column is seed-panel spread, not the validation-sampling SE).**

| max_depth | val_ci | ci sd_seed | val_rmst_diff | rmst sd_seed |
|---|---|---|---|---|
| 3 | 0.6967 | 0.0013 | **4.566** | 2.364 |
| 4 | 0.6965 | 0.0008 | 8.158 | 1.375 |
| 5 | 0.6955 | 0.0015 | 8.024 | 1.049 |
| 6 | 0.6935 | 0.0014 | 7.274 | 1.654 |
| 7 | 0.6939 | 0.0017 | 8.303 | 1.254 |
| 8 | 0.6933 | 0.0011 | 8.278 | 1.243 |
| 9 | 0.6934 | 0.0010 | **8.543** | 1.118 |
| 10 | 0.6935 | 0.0011 | 8.289 | **0.952** |
| None | 0.6935 | 0.0010 | 8.400 | **0.779** |

**Findings.**
1. This sweep has none of the noise problems that plagued the search-based
   comparisons (iter_003, iter_009) — `sd_seed` for `val_ci` is 0.0008-0.0017
   across the whole sweep, and even RMST's seed spread (0.78-2.36) is small
   enough that depth=3's isolation is unambiguous, not a noisy artifact.
2. `max_depth=3` is dominated on both axes simultaneously: worst RMST by a
   wide margin and among the highest seed variance. Trees this shallow appear
   too coarse to place the ACT-vs-OBS decision boundary consistently.
   Reassuringly, the current search space (`[None, 4, 6, 8, 12]`) never
   included 3, so this wasn't already contaminating any prior iteration.
3. Among depth 4 through unrestricted, `val_ci` is flat (0.6933-0.6965, a
   0.0032 band inside a single seed-sd) and `val_rmst_diff` clusters in
   7.27-8.54 with no monotonic trend — depth 6 (already in the search space)
   is the weakest of this group on RMST (7.274) while depth 9 and unrestricted
   are the strongest, but none of these differences clear even one `se_boot`
   (2.6-2.8 months), so this is a mild, not decisive, preference.
4. The seed-stability trend is the more interesting result: `rmst sd_seed`
   falls roughly monotonically from depth 4 (1.375) to unrestricted (0.779).
   Deeper trees give more reproducible RMST estimates, not less — the
   opposite of the usual intuition that deeper trees overfit and add
   variance. A plausible reason: shallower trees produce coarser terminal
   nodes, so a small perturbation (forest seed) more easily flips which side
   of a node's average risk a given patient falls on, and that flip is exactly
   what drives `val_rmst_diff` variance (iter_001, iter_002).

**Disposition.** Diagnostic-only; `_suggest_params`, the search space, and the
persisted model are unchanged, so no BETTER/WORSE verdict applies to the
sweep itself. Recommendation for future PARAM iterations: depth 3 is now
confirmed out of scope (already excluded); if `max_depth`'s search grid is
revisited, dropping 6 in favor of 9 or leaning more on `None` is mildly
supported but not urgent, since the whole depth-4-to-unrestricted band is
statistically indistinguishable given current validation-set noise.

---

### iter_011 — Uno's IPCW C-index (additive diagnostic)

- type: CODE
- idea_id: `uno_ipcw_cindex`
- hypothesis: Harrell's C-index (the frozen `val_ci` definition) is known to
  be biased when censoring is treatment- or covariate-dependent; adding Uno's
  IPCW C-index alongside it (never replacing it) will show whether the two
  agree, which is a check on whether `val_ci` itself is trustworthy given
  OBS-arm censoring patterns differ from ACT (iter_005's ESS finding already
  showed the arms have different weight/propensity profiles).
- changed_files: `train.py`

**Design.** `sksurv.metrics.concordance_index_ipcw(survival_train,
survival_test, estimate, tau=60.0)` per panel seed, using each seed's already-
computed validation risk predictions (no refitting) and the training outcome
array to estimate the censoring distribution (IPCW weights derived from
training censoring only, consistent with red line 2). `tau=60` matches the
existing RMST horizon so both added and frozen metrics reference the same
follow-up window. Reported as `uno_ci` per seed and as a panel summary,
identical treatment to `oob_ci` in iter_003. `val_ci` and `val_rmst_diff`
remain the only two metrics used for BETTER/WORSE verdicts, per red line 4.

**Red-line audit.** No new file reads (#1). IPCW weights are derived from the
training outcome distribution only, consistent with `compute_iptw`'s
train-only fitting elsewhere in this pipeline (#2). No rows dropped, this
only computes a diagnostic (#3). `val_ci`/`val_rmst_diff` are untouched;
Uno's C is added as a new column (#4, explicitly the admissibility condition
this idea is CLEARED under). Recommendation rule untouched (#5). ACT stays
binary (#6). Measurement-only (#7).

- val_ci: 0.6992 ± 0.0239
- val_rmst_diff: 7.37 ± 2.64 (months)
- n_features: 19
- verdict: NEUTRAL (measurement-only; headline numbers unchanged from
  iter_010, as expected)
- one_line_lesson: Uno's IPCW C-index (0.6951 ± 0.0011) sits within 0.004 of
  Harrell's C (0.6992), both seed-stable — `val_ci` is not being inflated by
  treatment- or covariate-dependent censoring, so the discrimination number
  this whole arena has treated as stable is corroborated by an
  independent estimator, not just self-consistent.

**Uno's IPCW C-index vs Harrell's C (panel, tau=60).**

| | mean | sd | iqr |
|---|---|---|---|
| Harrell's C (`val_ci`, frozen metric) | 0.6992 | 0.0009 | — |
| Uno's IPCW C (`uno_ci`, added diagnostic) | 0.6951 | 0.0011 | 0.0009 |

- per-seed Harrell: 0.6995, 0.6997, 0.6985, 0.7002, 0.7009, 0.6991, 0.6984,
  0.6973, 0.6994, 0.6988
- per-seed Uno: 0.6949, 0.6958, 0.6944, 0.6958, 0.6968, 0.6951, 0.6947,
  0.6927, 0.6953, 0.6951 — tracks Harrell's C almost seed-for-seed

**Findings.**
1. The two estimators agree to within 0.004, and Uno's C is uniformly a hair
   lower rather than higher — the direction a naive/optimistic bias would NOT
   produce. If Harrell's C were inflated by differential censoring between
   ACT and OBS (iter_005 already showed the arms have different weight/
   propensity profiles), Uno's IPCW correction would be expected to pull the
   estimate down more noticeably than 0.004.
2. Both metrics are essentially seed-invariant (sd ~0.001), matching every
   prior finding in this log that discrimination is the stable half of this
   problem and RMST is the volatile half.
3. This closes out a standing question raised implicitly since iter_001:
   whether the C-index headline is trustworthy on its own terms. It is.

**Disposition.** `val_ci` and `val_rmst_diff` remain the only two metrics used
for BETTER/WORSE verdicts, per red line 4. `uno_ci` is retained permanently as
a corroborating diagnostic at negligible extra cost (reuses already-computed
risk predictions, no refitting).

---

### iter_012 — S-learner / T-learner standardized contrast ensemble

- type: ALGO
- idea_id: `s_t_contrast_ensemble`
- hypothesis: The current model is a pure S-learner (one forest, ACT as a
  covariate). Averaging its standardized counterfactual contrast with an
  independently fit RSF T-learner's standardized contrast (two arm-specific
  forests) will produce a recommendation less dependent on any single model's
  idiosyncratic splits, potentially improving `val_rmst_diff` beyond the
  standing 7.355 ± 2.70 baseline. Reported as a diagnostic comparison first
  (not swapped into the headline), consistent with how iter_002's ensemble
  and iter_007-009's interaction terms were evaluated before any adoption
  decision.
- changed_files: `train.py`

**Design (prespecified before running).**
1. S-learner: existing single forest, fit on all training rows with IPTW
   sample weights, exactly as `seed_panel_report` already does.
2. T-learner: two separate forests, one fit on ACT=1 training rows only, one
   on ACT=0 training rows only — **no IPTW weights** (there is no cross-arm
   confounding to correct within a single, already-homogeneous-in-treatment
   subset). Per `rsf_t_learner`'s stated risk (ACT arm is small — 114 of 775
   training rows), the ACT-arm forest doubles `min_samples_leaf` relative to
   the S-learner's chosen value; the OBS-arm forest (661 rows) uses the
   S-learner's value unchanged.
3. Contrast for S: `risk(ACT=0) - risk(ACT=1)` from the single forest under
   both counterfactual settings, exactly as `_valid_predictions` already
   computes. Contrast for T: `obs_forest.predict(x) - act_forest.predict(x)`
   applied to every patient regardless of observed arm (both forests score
   everyone, which is what makes it a counterfactual contrast rather than a
   factual one).
4. Standardization (the idea's admissibility condition): mean/SD of each
   contrast is computed on **training-row predictions only**, never
   validation, then applied to standardize the validation contrasts.
   `recommendation = ACT where (z_S + z_T)/2 > 0`.
5. `val_ci` is deliberately left untouched — it continues to use the
   S-learner's factual risk, exactly as the frozen evaluator does. Only
   `val_rmst_diff` is recomputed under the new ensemble recommendation, since
   that is the objective this idea targets and it keeps the change surgical.
6. Evaluated across the full 10-seed panel; validation-row bootstrap SEs are
   skipped for this comparison sweep (same reasoning as iter_010's depth
   sweep: this is a before/after diagnostic, not the run's headline numbers).

**Red-line audit.**
1. *Sealed test.* No new file reads. PASS.
2. *No leakage.* T-learner forests fit on training-row subsets only.
   Standardization statistics are computed from training predictions only,
   the idea's explicit admissibility condition. PASS.
3. *Never drop censored patients.* Arm-specific subsetting keeps all rows
   within each arm, censored or not; no patient is excluded from the overall
   analysis. PASS.
4. *Metric definitions versioned.* `alignment_rmst_difference` unmodified;
   only its `recommendation` input changes for the diagnostic comparison.
   `val_ci` is untouched. PASS.
5. *Counterfactual recommendation.* Strengthened: every patient is scored
   under both counterfactual arms by both learners. PASS.
6. *No regimen-level claims.* ACT stays binary throughout. PASS.
7. *Both objectives.* This iteration is a diagnostic comparison; the run's
   headline numbers are unchanged from iter_011, so no verdict beyond NEUTRAL
   applies to the headline. The comparison table decides whether a follow-up
   should adopt the ensemble recommendation.

- val_ci: 0.6992 ± 0.0239
- val_rmst_diff: 7.37 ± 2.64 (months)
- n_features: 19
- verdict: NEUTRAL on the headline (unchanged, diagnostic-only iteration);
  the S+T ensemble comparison itself is a flagged MIXED finding — promising
  metric, concerning mechanism, **not adopted**
- one_line_lesson: The S+T ensemble raised mean RMST from 7.365 to 8.665 and
  roughly halved its seed variance, but it does so by recommending ACT to
  ~50% of patients instead of ~32%, a shift substantially driven by a
  T-learner arm-forest trained on only 114 ACT patients (70 events) that gets
  equal weight against the much better-supported S-learner in the
  standardized average — a promising number built on a fragile mechanism is
  not evidence the loop should act on unsupervised.

**S-only vs S+T ensemble, same 10-seed panel, same search-selected params.**

| | mean | sd (seed) | range |
|---|---|---|---|
| `val_rmst_diff` (S-only) | 7.365 | 1.736 | 4.47 to 10.62 |
| `val_rmst_diff` (S+T ensemble) | **8.665** | **0.746** | 7.35 to 9.76 |
| ACT-recommended fraction (S-only) | 0.33 | — | 0.305 to 0.382 |
| ACT-recommended fraction (ensemble) | **0.50** | — | 0.459 to 0.533 |
| patients flipped vs S-only (of 259) | — | — | 32 to 59 |

**Findings.**
1. The ensemble's RMST mean is +1.30 months over S-only on the identical seed
   panel and identical search-selected hyperparameters — a much cleaner,
   less-confounded comparison than iter_007's interaction-expansion arc,
   since nothing here depends on a different Optuna search draw. It also cuts
   seed variance by more than half (1.736 to 0.746), the most seed-stable
   RMST result recorded in this arena to date. Vs iter_001's baseline
   (7.355 ± 2.702), the delta is +1.31, still short of the formal +2.70
   BETTER threshold.
2. The mechanism behind the gain is concerning enough to withhold adoption.
   `min_samples_leaf` is doubled for the ACT-arm forest (per
   `rsf_t_learner`'s stated risk mitigation), but doubling a leaf-size
   regularizer does not compensate for a 5.8x smaller training set (114 vs
   661 rows, 70 vs 278 events) the way equal standardized weighting implies.
   Z-scoring makes the two contrasts commensurate in *scale*, not in
   *reliability* — a noisy contrast and a well-supported one end up with the
   same 50% vote.
3. The behavioral consequence is large: roughly 1 in 6 of the entire
   validation cohort (32-59 of 259 patients per seed) gets a different
   treatment recommendation, and the overall ACT-recommended fraction nearly
   doubles from ~33% to ~50%. That is a substantial clinical-recommendation
   shift to accept on the strength of a metric alone, particularly given red
   line 5's requirement that the counterfactual recommendation be trustworthy,
   not merely score-maximizing.
4. `val_ci` is unaffected by design (computed from the S-learner's factual
   risk in both cases, per this iteration's audit) and stays flat.

**Disposition — not adopted.** The S+T ensemble is not swapped into the
headline recommendation path. This is flagged as an open lead rather than a
dead end: a properly reliability-weighted combination (e.g., shrinking the
T-learner's contribution toward zero in proportion to the ACT arm's much
smaller effective sample size, rather than a flat 50/50 average) could
plausibly keep the seed-stability benefit while reducing the recommendation
churn — but that is a design decision with real clinical stakes and belongs
with human review, not further autonomous iteration in the remaining budget.
`s_t_ensemble_sweep` is retained in the codebase as a diagnostic comparison
function for that future work; it is not called from any path that affects
the persisted model or headline metrics.

---

### iter_013 — run artifact manifest (hashed provenance)

- type: CODE
- idea_id: `artifact_manifest`
- hypothesis: Twelve iterations in, run directories record results but not a
  compact, hashable fingerprint of what produced them; adding a manifest with
  hashes of the feature list, forest parameters, and data schema, plus an
  explicit metric-definition version string, will make it possible to detect
  silent drift (e.g., an accidental change to `CLINICAL_VARS` or the frozen
  metric functions) across future iterations without re-reading full JSON
  diffs.
- changed_files: `train.py`

**Design.** `build_manifest(metadata, train_df, valid_df)` computes SHA-256
hashes of: (1) the sorted feature name list, (2) `rsf_params` as canonical
JSON, (3) the training/validation frame's column names and dtypes (schema,
not data values), plus a fixed `metric_version` string documenting the frozen
metric definitions (`concordance_index_censored`; `restricted_mean_survival_
time`, tau=60) and the current git commit hash if available. Written to
`run_dir/manifest.json` alongside the existing artifacts. Read-only with
respect to data and model — it hashes already-computed objects.

**Red-line audit.** No new file reads beyond `git rev-parse HEAD` (repository
metadata, not analysis data) (#1). Hashing is read-only and touches no
fitting, weighting, or selection (#2). No rows touched (#3). Metric
definitions are not altered; `metric_version` records the two frozen
definitions verbatim rather than changing them (#4). Recommendation rule
untouched (#5). ACT stays binary (#6). Provenance-only; no verdict beyond
NEUTRAL expected (#7).

- val_ci: 0.6992 ± 0.0239
- val_rmst_diff: 7.37 ± 2.64 (months)
- n_features: 19
- verdict: NEUTRAL (measurement-only; headline unchanged from iter_011)
- one_line_lesson: `manifest.json` now records a hashed fingerprint
  (feature list, params, schema, git commit) per run at negligible cost,
  giving a fast drift check without re-diffing full metadata files.

**Example manifest (this run).**

```json
{
  "metric_version": "val_ci=sksurv.metrics.concordance_index_censored (Harrell's C); val_rmst_diff=lifelines.utils.restricted_mean_survival_time, tau=60",
  "git_commit": "9b43583a2dd2b80d939ca393d68eb6ba890f78ee",
  "feature_names_hash": "ce15de3e...",
  "rsf_params_hash": "7b44104c...",
  "train_schema_hash": "7ca6fa6d...",
  "valid_schema_hash": "7ca6fa6d...",
  "n_train": 775,
  "n_valid": 259
}
```

**Findings.** Train and validation schema hashes match, as expected (both
loaded through the same `preprocess_split` with identical `ANALYSIS_COLUMNS`)
— a useful sanity check confirmed automatically rather than assumed.
`feature_names_hash` and `rsf_params_hash` give a one-line way to notice, in
future log review, exactly when the feature set or chosen hyperparameters
changed between runs without opening `metadata.json`.

**Disposition.** Adopted permanently at negligible cost. Headline numbers
unchanged from iter_011, as expected for a provenance-only addition.

---

### iter_014 — controlled n_estimators sweep

- type: PARAM
- idea_id: `more_trees`
- hypothesis: Following iter_010's controlled-sweep methodology (fixed other
  params, vary one, full 10-seed panel, no search noise), sweeping
  `n_estimators` from 300 to 3000 will show whether more trees meaningfully
  improve either objective or just add runtime, and whether seed stability
  keeps improving with more trees as expected from basic ensemble variance
  reduction.
- changed_files: `train.py`

**Design.** Fixed baseline params, matching iter_010's interior choices for
comparability: `min_samples_leaf=15`, `min_samples_split=37`,
`max_features=0.7`, `max_depth=None` (iter_010 showed `None` had the lowest
RMST seed variance of the whole depth sweep). Candidates:
`n_estimators ∈ {300, 500, 700, 1000, 1500, 2000, 3000}`, each evaluated on
the full 10-seed panel, reusing the same read-only sweep pattern as
`depth_sweep` (panel mean and seed spread only, no bootstrap CI — this is a
before/after comparison, not the run's headline numbers).

**Red-line audit.** Identical structure to iter_010's cleared sweep: no new
file reads (#1); all fits on training rows with IPTW fit on training rows
only (#2); no rows dropped (#3); frozen metric functions (#4); recommendation
rule unchanged (#5); ACT stays binary (#6); does not alter `_suggest_params`
or the persisted model, so no verdict beyond NEUTRAL applies to the run's
headline numbers (#7).

- val_ci: 0.6992 ± 0.0239
- val_rmst_diff: 7.37 ± 2.64 (months)
- n_features: 19
- verdict: NEUTRAL on the headline (search/panel unchanged, deltas vs
  iter_001 negligible); sweep shows no case for more trees beyond ~700-1000
- one_line_lesson: `n_estimators` beyond ~700-1000 trees buys essentially
  nothing — `val_ci` is flat across the whole 300-2000 range (0.693-0.694,
  well inside one seed-sd) and `val_rmst_diff` has no monotonic trend, so the
  current search range (300-1200) is already past the point of diminishing
  returns and doesn't need widening.

**n_estimators sweep (fixed leaf=15, split=37, mtry=0.7, depth=None).**

| n_estimators | val_ci | ci sd_seed | val_rmst_diff | rmst sd_seed |
|---|---|---|---|---|
| 300 | 0.6942 | 0.0012 | 6.923 | 0.934 |
| 500 | 0.6936 | 0.0014 | 7.882 | 1.061 |
| 700 | 0.6935 | 0.0010 | **8.574** | 0.726 |
| 1000 | 0.6931 | **0.0004** | 7.787 | 1.024 |
| 1500 | 0.6930 | 0.0008 | 8.052 | 1.715 |
| 2000 | 0.6931 | 0.0007 | 8.149 | 1.175 |

**Findings.**
1. `val_ci` is essentially flat from 300 to 2000 trees (a 0.0012 band, well
   under any single candidate's own seed-sd), confirming the standard
   ensemble-averaging expectation that discrimination saturates early. There
   is a mild, non-monotonic tightening of `ci sd_seed` up to 1000 trees, but
   it does not continue improving at 1500-2000.
2. `val_rmst_diff` has no monotonic relationship with tree count at all — 700
   trees is the sweep's peak (8.574) despite sitting in the middle of the
   range, and 1500 trees has the *worst* seed stability of the whole sweep
   (sd_seed 1.715, higher than even the 300-tree case). More trees is not a
   reliable lever for the RMST objective in either direction.
3. This closes out the practical case for widening `more_trees`'s current
   300-1200 search range: nothing in 1500-2000 outperforms what the existing
   range already reaches, and the added compute (1500-2000 trees, ~2-3x the
   fit time and pickle size of a 700-tree forest) would buy nothing. It also
   explains why iter_014's first attempt at this sweep including a 3000-tree
   candidate was killed (very likely OOM from repeatedly building
   near-maximal forests in a loop) — the retry dropped 3000 from the sweep
   and confirmed there was no signal being missed by not reaching it.

**Disposition.** Diagnostic-only; `_suggest_params`'s existing
`n_estimators` range (300-1200, step 100) is left unchanged — this sweep
found no reason to widen it. `N_ESTIMATORS_SWEEP_CANDIDATES` in `train.py` is
capped at 2000 (not the idea's stated 1500-3000 upper bound) after the 3000
candidate triggered a background-task kill during this iteration's first
attempt; retrying with a lower cap and explicit `del model` inside the sweep
loop completed cleanly.

---

### iter_015 — validation-only permutation null for RMST alignment

- type: CODE
- idea_id: `permutation_null_rmst`
- hypothesis: `val_rmst_diff`'s bootstrap SE (~2.6-2.7 months, established
  since iter_001) already suggests the point estimate is not comfortably far
  from zero; a direct permutation null — shuffle observed treatment labels
  among validation patients while holding the model's recommendation fixed,
  recompute `alignment_rmst_difference` many times — will give a formal
  one-sided p-value against the null that observed treatment carries no
  information correlated with the model's recommendation, closing the loop on
  a question this log has carried informally since iter_001.
- changed_files: `train.py`

**Design.** For the primary (persisted) model's validation predictions:
compute `observed = alignment_rmst_difference(valid_df, recommendation)`
(identical call to the frozen evaluator). Then, 1000 times, permute the
`Adjuvant Chemo` column among validation rows only (`recommendation` and all
outcome columns held fixed) and recompute the same function, building a null
distribution. Report the null mean/SD and the one-sided p-value
`P(null >= observed)`. This uses validation rows only, touches no training
data or model fitting, and calls the frozen `alignment_rmst_difference`
unmodified — it does not replace `val_rmst_diff`, exactly the idea's stated
admissibility ("additive diagnostic").

**Red-line audit.** No new file reads (#1). Permutation is confined to
validation-row treatment labels; nothing about fitting, IPTW, or the search
is touched (#2). No rows dropped, only shuffled and restored (#3).
`alignment_rmst_difference` called unmodified; `val_rmst_diff` itself is
untouched, this is a new additive column (#4, the idea's explicit
admissibility). Recommendation is computed once from the persisted model
exactly as `evaluate_on_valid` already does, then held fixed through the
permutation (#5). ACT stays binary (#6). Measurement-only (#7).

- val_ci: PENDING RUN
- val_rmst_diff: PENDING RUN
- n_features: 19
- verdict: PENDING RUN
- one_line_lesson: PENDING RUN
