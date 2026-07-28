# Red Lines

1. **Test set is sealed.** `affyfRMATest.csv` is read exactly once, at the very end, by the human. The agent's `prepare.py` must not load, reference, or expose test data in any form. Any code path that opens the test file during the loop is a violation.
2. **No leakage in any preprocessing step.** IPTW propensity models, scalers, encoders, and feature selection are fit on the bootstrap's training rows only. Validation rows pass through transforms fitted on training rows.
3. **Never drop censored patients.** All n in the analytic cohort are used.
4. **Metric definitions are versioned.** Harrell's C from `sksurv.metrics.concordance_index_censored`, RMST at tau=60 months from `lifelines.utils.restricted_mean_survival_time`. If the agent wants to *add* a metric, it adds it as an additional column in `log.md`, never replacing the existing two.
5. **Treatment recommendation must be counterfactual.** Predict risk under ACT=1 and ACT=0 for every patient; recommend the lower-risk arm. Do not regress on observed outcomes within treatment arms as a shortcut.
6. **No regimen-level claims.** ACT is binary. Don't infer or assume cisplatin vs. carboplatin, doublet vs. single-agent, dose intensity, or schedule effects.
7. **Improvement claims require both objectives to hold.** A new candidate is "BETTER" only if val_ci AND val_rmst_diff each beat the previous best by more than the bootstrap SE / IQR of the metric. One-objective wins are MIXED, not BETTER.
8. **No hardcoded predictions, no oracle features, no test-set peeking.** If the agent finds itself "fixing" the metric to climb the score, that's a violation.
