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

- val_ci: PENDING RUN
- val_rmst_diff: PENDING RUN
- n_features: 19
- verdict: PENDING RUN
- one_line_lesson: PENDING RUN
