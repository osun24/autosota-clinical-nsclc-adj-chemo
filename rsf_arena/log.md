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

- val_ci: PENDING RUN
- val_rmst_diff: PENDING RUN
- n_features: PENDING RUN
- verdict: PENDING RUN
- one_line_lesson: PENDING RUN
