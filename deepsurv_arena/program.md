# DeepSurv clinical-only arena

Optimize validation Harrell C-index and 5-year RMST treatment-alignment
difference jointly. Use only `clinicalTrain.csv` and
`clinicalValidation.csv`; the test set remains sealed.

All scaling and IPTW estimation must be fit without validation or test
leakage. Treatment recommendations must compare counterfactual predictions
under ACT and observation. A candidate is BETTER only when both objectives
improve beyond bootstrap uncertainty.

The arena is CPU-safe and uses CUDA when available. Do not run
`../finalize-deepsurv.py` or `../finalize-all.py` during development.
