# Project: NSCLC adjuvant chemo HTE — DeepSurv arena

## Goal
Improve the `train.py` pipeline along TWO objectives jointly:
- `val_ci` (Val Harrell's C-index, higher better)
- `val_rmst_diff` (Val 5-yr RMST diff under counterfactual recommendation, higher better)
Both must improve beyond bootstrap uncertainty for a candidate to count as BETTER.

## Hard Rules
See `red_lines.md`. The test set remains sealed; `prepare.py` and `train.py` must only load Train and Validation.

## Runtime Note
This arena is GPU-friendly but CPU-safe. On this machine, base Python reports CUDA unavailable, so the default budget is conservative. If a CUDA GPU is available in a later session, the human may raise `DEEPSURV_ARENA_EPOCHS`, `DEEPSURV_ARENA_N_TRIALS`, and `DEEPSURV_ARENA_BOOTSTRAPS` before launch.

If Colab is the only GPU source, use `../colab_bridge/local_colab_queue.py` locally and `../colab_bridge/colab_gpu_worker.py` in a manually started Colab runtime. Colab is a worker only; the local loop still chooses ideas, edits code, and appends `log.md` after collecting the returned artifact. The Colab data directory must contain train/validation only.

## Per-Iteration Workflow
1. Read `log.md`. Identify the last 3 iterations and their `type` (PARAM | CODE | ALGO).
2. If all 3 most recent COMPLETED iterations were PARAM, propose a CODE or ALGO idea this iteration.
3. Pick one CLEARED idea from the idea library below, or generate a new one.
4. Write an idea entry to `log.md` with type, hypothesis, expected effect, and a red-line audit BEFORE editing `train.py`.
5. `git add . && git commit -m "iter_NNN_pre: <idea_id>"` to snapshot pre-state.
6. Edit `train.py`. Run `python deepsurv_arena/train.py`. Wall clock budget: 25 min on CPU unless the human approves a GPU budget.
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
- `clinical_interaction_inputs`: Add a prespecified, hierarchical set of covariate-by-ACT and clinically interpretable covariate-by-covariate inputs. Risk: medium because the network can overfit redundant terms. Admissibility: CLEARED with a fixed interaction budget.
- `shared_trunk_dual_head`: Learn a shared clinical representation with separate ACT and OBS risk heads so the small ACT arm borrows prognostic structure. Risk: medium to high because the survival loss must combine arm-specific risk sets correctly. Admissibility: CLEARED after a written loss audit.
- `t_learner_deepsurv`: Fit separate, arm-regularized DeepSurv models for ACT and OBS and compare arm-specific risks. Risk: high because the ACT subgroup is small. Admissibility: CLEARED after an arm-level event-count and calibration audit.
- `treatment_gated_network`: Use a shared prognostic trunk plus a constrained treatment-effect branch driven by pretreatment covariates. Risk: high because identifiability and regularization choices affect the contrast. Admissibility: CLEARED only with an explicit architecture and counterfactual audit.
- `s_t_contrast_ensemble`: Average standardized treatment-risk contrasts from the current S-learner and a DeepSurv T-learner. Risk: medium. Admissibility: CLEARED if contrast scaling is trained without validation outcomes.
- `cross_fitted_iptw`: Produce out-of-fold propensity weights inside every bootstrap before neural-network fitting. Risk: medium due to sparse ACT folds. Admissibility: CLEARED with treatment-by-event stratification.
- `recommendation_stability_ensemble`: Average counterfactual contrasts over a fixed seed panel and report agreement rates. Risk: low to medium. Admissibility: CLEARED when the mean contrast defines the recommendation.

### CODE
- `train_only_early_stopping`: Split each bootstrap internally by treatment and event, fit scaling on its training portion, and stop on its held-out portion. Risk: low to medium because the ACT stopping set is small. Admissibility: CLEARED with minimum-cell guards.
- `breslow_tie_tests`: Add unit tests comparing the weighted Breslow loss with hand-calculated tied and censored examples. Risk: low. Admissibility: CLEARED.
- `seed_panel_report`: Save per-seed metrics, counterfactual contrasts, and recommendation agreement without changing the objectives. Risk: low. Admissibility: CLEARED.
- `gradient_health_report`: Record non-finite losses, gradient norms, and convergence summaries for every trial. Risk: low. Admissibility: CLEARED.
- `permutation_null_rmst`: Add a validation-only treatment-label permutation null for RMST alignment. Risk: low. Admissibility: CLEARED as an additive diagnostic.
- `uno_ipcw_cindex`: Add Uno's IPCW C-index alongside Harrell's C-index without replacing either objective. Risk: low. Admissibility: CLEARED.
- `artifact_manifest`: Hash the clinical feature list, network parameters, transform arrays, data schema, and metric version. Risk: low. Admissibility: CLEARED.

### PARAM
- `compact_architectures`: Compare linear, 8-unit, 16-unit, 16-8, and 32-16 networks before considering wider models for the 19-feature input. Risk: low. Admissibility: CLEARED.
- `regularization_grid`: Jointly tune dropout `[0.0, 0.5]` and weight decay `[1e-6, 1e-2]`, favoring simpler models when objectives are statistically tied. Risk: low. Admissibility: CLEARED.
- `learning_rate_schedule`: Compare fixed AdamW learning rates with a training-only plateau or cosine schedule. Risk: low. Admissibility: CLEARED.
- `l1_first_layer`: Add a small L1 penalty to the first layer to encourage sparse use of clinical covariates. Risk: low to medium. Admissibility: CLEARED.
- `epoch_budget`: Compare 50-300 epochs only after train-only early stopping exists; do not select epochs directly on the shared validation set. Risk: medium. Admissibility: CLEARED with that prerequisite.
- `iptw_clip_sweep`: Compare prespecified clipping ranges while reporting effective sample size and choosing from training diagnostics only. Risk: medium because clipping changes the weighted estimand. Admissibility: CLEARED with the diagnostic restriction.

## Finalization
Do not run `../finalize-deepsurv.py` or `../finalize-all.py` during the autonomous loop. They are HUMAN ONLY and may load `affyfRMATest.csv` after the user adds that file locally.
