# Clinical-only NSCLC adjuvant chemotherapy modeling

This repository evaluates treatment-recommendation models built only from
clinical covariates. The analysis schema contains treatment, age, sex, stage,
histology, race, smoking status, overall-survival status, and overall-survival
time.

## Build the analysis datasets

Place the source train and validation CSVs in the repository root, then run:

```bash
python3 drop-nonclinical.py
```

This writes `clinicalTrain.csv` and `clinicalValidation.csv`. The conversion
uses an explicit 21-column allowlist, rejects missing required columns, drops
the source row identifier, and reports how many non-analysis columns it
removed. Alternate paths can be supplied with `--train`, `--validation`,
`--train-out`, and `--validation-out`.

## Model arenas

- `xgb_arena/` fits an XGBoost-Cox T-learner using the 18 pretreatment
  covariates.
- `rsf_arena/` fits a Random Survival Forest using all 19 clinical covariates.
- `deepsurv_arena/` fits DeepSurv using all 19 clinical covariates.

Each arena exposes `run() -> dict`, optimizes validation C-index and 5-year
RMST treatment-alignment difference, and writes new artifacts below its
`runs/` directory.

```bash
python3 xgb_arena/train.py
python3 rsf_arena/train.py
python3 deepsurv_arena/train.py
```

The test set remains sealed during model development. After choosing runs and
adding `affyfRMATest.csv` locally, a human may perform the single final
evaluation:

```bash
python3 finalize.py --xgb-run-dir xgb_arena/runs/<chosen_run>
python3 finalize-rsf.py --rsf-run-dir rsf_arena/runs/<chosen_run>
python3 finalize-deepsurv.py --deepsurv-run-dir deepsurv_arena/runs/<chosen_run>
```

## Colab DeepSurv worker

The Colab data directory must contain only `clinicalTrain.csv` and
`clinicalValidation.csv`. Never place the sealed test CSV in that directory.
See `colab_bridge/README.md` for queue commands.
