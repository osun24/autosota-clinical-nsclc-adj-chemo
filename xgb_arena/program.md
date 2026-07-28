# Project: NSCLC adjuvant chemo HTE — XGBoost arena

## Goal
Improve the `train.py` pipeline along TWO objectives jointly:
- `val_ci` (Val Harrell's C-index, higher better)
- `val_rmst_diff` (Val 5-yr RMST diff under counterfactual recommendation, higher better)
Both must improve beyond bootstrap uncertainty for a candidate to count as BETTER.

## Hard Rules
1. **Test set is sealed.** `affyfRMATest.csv` is read exactly once, at the very end, by the human. The agent's `prepare.py` must not load, reference, or expose test data in any form. Any code path that opens the test file during the loop is a violation.
2. **No leakage in any preprocessing step.** IPTW propensity models, scalers, encoders, feature selection, and calibration are fit on the bootstrap's training rows only. Validation rows pass through transforms fitted on training rows.
3. **Never drop censored patients.** All n in the analytic cohort are used.
4. **Metric definitions are versioned.** Harrell's C from `sksurv.metrics.concordance_index_censored`, RMST at tau=60 months from `lifelines.utils.restricted_mean_survival_time`. If the agent wants to *add* a metric (e.g., Uno's IPCW C, C-for-benefit from Van Klaveren 2018), it adds it as an additional column in `log.md`, never replacing the existing two.
5. **Treatment recommendation must be counterfactual.** Predict risk under ACT=1 and ACT=0 for every patient; recommend the lower-risk arm. Do not regress on observed outcomes within treatment arms as a shortcut.
6. **No regimen-level claims.** ACT is binary. Don't infer or assume cisplatin vs. carboplatin, doublet vs. single-agent, dose intensity, or schedule effects.
7. **Improvement claims require both objectives to hold.** A new candidate is "BETTER" only if val_ci AND val_rmst_diff each beat the previous best by more than the bootstrap SE / IQR of the metric. One-objective wins are MIXED, not BETTER.
8. **No hardcoded predictions, no oracle features, no test-set peeking.** If the agent finds itself "fixing" the metric to climb the score, that's a violation.

## Per-Iteration Workflow
1. Read `log.md`. Identify the last 3 iterations and their `type` (PARAM | CODE | ALGO).
2. If all 3 most recent COMPLETED iterations were PARAM, you MUST propose a CODE or ALGO idea this iteration. This is the Leap Path rule from AutoSOTA.
3. Pick one CLEARED idea from the idea library below, or generate a new one.
4. Write an idea entry to `log.md` with type, hypothesis, expected effect, and a red-line audit BEFORE editing `train.py`.
5. `git add . && git commit -m "iter_NNN_pre: <idea_id>"` to snapshot pre-state.
6. Edit `train.py`. Run `python xgb_arena/train.py`. Wall clock budget: 25 min.
7. On completion, append the result to `log.md` in the schema below.
8. If verdict == WORSE on both objectives, `git revert HEAD` and choose a different idea. If MIXED, leave the change in place but flag it; pursue at most 3 follow-up iterations to debug before rolling back (the AutoSOTA honeymoon period).

## log.md Entry Schema
Use this exactly.

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
- `clinical_s_learner`: Fit one XGBoost-Cox model with ACT, all pretreatment covariates, and a controlled set of covariate-by-ACT terms; obtain recommendations by toggling ACT. Risk: medium because explicit interactions can overfit. Admissibility: CLEARED with a fixed interaction budget.
- `shared_prognostic_plus_arm_models`: Fit a shared clinical prognostic model, then fit treatment-arm models to residual risk so the small ACT arm borrows common structure. Risk: medium because the offset construction must respect survival risk sets. Admissibility: CLEARED after a written loss-function audit.
- `s_t_contrast_ensemble`: Average standardized treatment-risk contrasts from the current T-learner and a clinical S-learner. Risk: medium because contrast scales may differ. Admissibility: CLEARED if all component models and contrast scaling are trained without validation outcomes.
- `arm_specific_capacity`: Allow the smaller ACT arm to use shallower trees or stronger regularization than the OBS arm. Risk: low to medium. Admissibility: CLEARED.
- `cross_fitted_iptw`: Estimate propensity scores out of fold within each bootstrap and use out-of-fold weights for model fitting. Risk: medium because fold construction can become unstable in the ACT arm. Admissibility: CLEARED if folds are stratified by treatment and event.
- `clinical_interaction_hierarchy`: Add only clinically interpretable two-way terms, such as stage-by-age, stage-by-histology, and smoking-by-histology, with main effects always retained. Risk: medium. Admissibility: CLEARED with a prespecified candidate list.
- `recommendation_stability_ensemble`: Fit a fixed panel of bootstrap models and recommend ACT only when the mean contrast favors ACT; report the fraction of models agreeing. Risk: medium because an abstention policy would alter the estimand. Admissibility: CLEARED for mean-contrast recommendations; abstention requires a separate protocol.

### CODE
- `stratified_refit_split`: Stratify the internal refit/early-stopping split by treatment arm and event status, with explicit minimum-cell checks. Risk: low. Admissibility: CLEARED.
- `arm_bootstrap_guards`: Require both events and censored observations within each treatment arm before accepting a bootstrap sample. Risk: low. Admissibility: CLEARED.
- `permutation_null_rmst`: Add a validation-only treatment-label permutation null for RMST alignment and report the observed-minus-null effect. Risk: low. Admissibility: CLEARED as an additive diagnostic.
- `uno_ipcw_cindex`: Add Uno's IPCW C-index alongside Harrell's C-index without replacing either primary objective. Risk: low. Admissibility: CLEARED.
- `recommendation_stability_report`: Save per-patient bootstrap recommendation frequency and summarize instability without using validation outcomes to choose a threshold. Risk: low. Admissibility: CLEARED.
- `artifact_manifest`: Hash the clinical feature list, model parameters, data schema, and metric version in every run directory. Risk: low. Admissibility: CLEARED.
- `bootstrap_finalize_ci`: At human-only finalization, compute percentile confidence intervals for sealed-test metrics without changing the chosen model. Risk: low. Admissibility: CLEARED.

### PARAM
- `shallow_tree_focus`: Restrict `max_depth` to 1-3 and expand `min_child_weight` upward to suit the 18-feature clinical input. Risk: low. Admissibility: CLEARED.
- `slow_learning_schedule`: Explore eta in `[0.003, 0.08]` with a larger boosting-round ceiling and proportional early stopping. Risk: low but slower. Admissibility: CLEARED.
- `clinical_column_sampling`: Compare full-column fitting against mild `colsample_bytree` regularization in `[0.7, 1.0]`. Risk: low. Admissibility: CLEARED.
- `stronger_penalties`: Expand L1/L2 penalties for the small ACT arm and shallow-tree models. Risk: low. Admissibility: CLEARED.
- `arm_weight_clip`: Compare prespecified IPTW clipping ranges such as `[0.2, 5]`, `[0.1, 10]`, and `[0.05, 20]`, reporting effective sample size for each. Risk: medium because clipping changes the weighted estimand. Admissibility: CLEARED only when the choice is made on training diagnostics.
- `bootstrap_budget`: Increase bootstrap replicates before increasing Optuna trials so trial selection reflects uncertainty. Risk: low but slower. Admissibility: CLEARED.

## Anti-Stagnation
If you have gone 5 iterations without a BETTER verdict on both objectives, explicitly try an ALGO move from the library. If the library is exhausted, read the latest survival-HTE literature (causal survival forest, BCF survival, deep counterfactual survival models) and propose a new ALGO entry, with red-line audit, before implementing.

## Finalization
Do not run `../finalize.py` during the autonomous loop. It is HUMAN ONLY and is allowed to load `affyfRMATest.csv` once the user has added that file locally.
