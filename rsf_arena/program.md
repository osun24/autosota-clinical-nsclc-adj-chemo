# Project: NSCLC adjuvant chemo HTE — Random Survival Forest arena

## Goal
Improve the `train.py` pipeline along TWO objectives jointly:
- `val_ci` (Val Harrell's C-index, higher better)
- `val_rmst_diff` (Val 5-yr RMST diff under counterfactual recommendation, higher better)
Both must improve beyond bootstrap uncertainty for a candidate to count as BETTER.

## Hard Rules
See `red_lines.md`. The test set remains sealed; `prepare.py` and `train.py` must load only `clinicalTrain.csv` and `clinicalValidation.csv`.

## Per-Iteration Workflow
1. Read `log.md`. Identify the last 3 iterations and their `type` (PARAM | CODE | ALGO).
2. If all 3 most recent COMPLETED iterations were PARAM, propose a CODE or ALGO idea this iteration.
3. Pick one CLEARED idea from the idea library below, or generate a new one.
4. Write an idea entry to `log.md` with type, hypothesis, expected effect, and a red-line audit BEFORE editing `train.py`.
5. `git add . && git commit -m "iter_NNN_pre: <idea_id>"` to snapshot pre-state.
6. Edit `train.py`. Run `python rsf_arena/train.py`. Wall clock budget: 25 min.
7. On completion, append the result to `log.md` using the schema below.
8. WORSE on both objectives should be reverted. MIXED may stay for at most 3 follow-up iterations.

## log.md Entry Schema
```markdown
### iter_NNN — <short_title>
- type: PARAM | CODE | ALGO
- idea_id: <slug>
- hypothesis: <one sentence>
- changed_files: <list>
- val_ci: 0.XXXX ± 0.XXXX
- val_rmst_diff: X.XX ± X.XX (months)
- n_features: <int>
- verdict: BETTER | WORSE | MIXED
- one_line_lesson: <text>
```

## Idea Library

### ALGO
- `clinical_interaction_expansion`: Add a prespecified, hierarchical set of covariate-by-ACT and clinically interpretable covariate-by-covariate terms before fitting the forest. Risk: medium because redundant terms can distort split selection. Admissibility: CLEARED with a fixed interaction budget.
- `rsf_t_learner`: Fit separate forests for ACT and OBS, using stronger regularization for the smaller ACT arm, then compare arm-specific risk predictions. Risk: high because the ACT sample is small. Admissibility: CLEARED after an arm-level event-count audit.
- `honest_rsf_split`: Use disjoint training subsamples for split selection and terminal-node estimation where supported, or emulate honesty with repeated sample splitting and aggregation. Risk: medium and implementation-heavy. Admissibility: CLEARED after verifying that validation data never informs the split.
- `balanced_arm_bootstrap`: Draw training bootstraps within treatment-by-event strata, then correct weights so the fitted estimand remains tied to the original cohort. Risk: medium because naive balancing changes the target population. Admissibility: CLEARED only with an explicit reweighting derivation.
- `s_t_contrast_ensemble`: Average standardized treatment-risk contrasts from the clinical S-learner forest and an RSF T-learner. Risk: medium. Admissibility: CLEARED if scaling is fit on training predictions only.
- `cross_fitted_iptw`: Generate out-of-fold propensity scores within each bootstrap before fitting the forest. Risk: medium because sparse strata can destabilize folds. Admissibility: CLEARED with treatment-by-event stratification.
- `recommendation_stability_forest`: Aggregate counterfactual contrasts across a fixed seed panel and report both the mean contrast and agreement rate. Risk: low to medium. Admissibility: CLEARED when the final recommendation uses the prespecified mean contrast.

### CODE
- `oob_diagnostics`: Record out-of-bag discrimination and compare it with validation discrimination without using the comparison to redefine the primary objectives. Risk: low. Admissibility: CLEARED.
- `seed_panel_report`: Save per-seed validation metrics, recommendation agreement, and aggregate uncertainty in run metadata. Risk: low. Admissibility: CLEARED.
- `effective_sample_size`: Report IPTW effective sample size overall and by treatment arm, plus weight quantiles. Risk: low. Admissibility: CLEARED.
- `permutation_null_rmst`: Add a validation-only treatment-label permutation null for RMST alignment. Risk: low. Admissibility: CLEARED as an additive diagnostic.
- `uno_ipcw_cindex`: Add Uno's IPCW C-index alongside Harrell's C-index without replacing either objective. Risk: low. Admissibility: CLEARED.
- `artifact_manifest`: Hash the clinical feature list, forest parameters, data schema, and metric version in every run directory. Risk: low. Admissibility: CLEARED.
- `bootstrap_finalize_ci`: At human-only finalization, compute confidence intervals for sealed-test metrics without changing model selection. Risk: low. Admissibility: CLEARED.

### PARAM
- `leaf_size_sweep`: Compare conservative `min_samples_leaf` values from 5 to 30; keep values below 5 out of scope unless an uncertainty audit justifies them. Risk: medium. Admissibility: CLEARED.
- `clinical_mtry`: Focus `max_features` on `[0.4, 1.0]`, appropriate for the compact clinical feature set. Risk: low. Admissibility: CLEARED.
- `depth_regularization`: Compare bounded depths of 3-10 with unrestricted depth while holding leaf size fixed. Risk: low. Admissibility: CLEARED.
- `more_trees`: Increase `n_estimators` to 1500-3000 only after fixing the other parameters, then verify seed stability. Risk: low but slow. Admissibility: CLEARED.
- `split_size_coordination`: Tune `min_samples_split` jointly with `min_samples_leaf` so invalid or redundant combinations are excluded. Risk: low. Admissibility: CLEARED.
- `iptw_clip_sweep`: Compare prespecified clipping ranges and report effective sample size before selecting a range using training diagnostics only. Risk: medium because clipping changes the weighted estimand. Admissibility: CLEARED with the diagnostic restriction.

## Finalization
Do not run `../finalize-rsf.py` or `../finalize-all.py` during the autonomous loop. They are HUMAN ONLY and may load `affyfRMATest.csv` after the user adds that file locally.
