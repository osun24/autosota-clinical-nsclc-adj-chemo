"""Frozen clinical-only data and validation utilities for XGBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from clinical_data import (
    CLINICAL_VARS,
    PRETREATMENT_VARS,
    TRAIN_CSV,
    VALID_CSV,
    alignment_rmst_difference,
    bootstrap_resample_df,
    build_matrix,
    cindex,
    clinical_columns,
    compute_iptw,
    load_train_valid,
    preprocess_split,
)

CLIN_FEATS_PRETX = PRETREATMENT_VARS


def pack_cox_labels(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=np.float32)
    event = np.asarray(event, dtype=int)
    return np.where(event == 1, time, -time).astype(np.float32)


def slice_booster_to_best_iteration(
    booster: xgb.Booster, best_ntree: int
) -> xgb.Booster:
    return booster[: int(best_ntree)]


def predict_xgb_risk(
    booster: xgb.Booster,
    matrix: np.ndarray,
    feature_names: list[str],
    best_ntree: int,
) -> np.ndarray:
    dmatrix = xgb.DMatrix(
        matrix.astype(np.float32), feature_names=feature_names
    )
    return booster.predict(
        dmatrix, iteration_range=(0, int(best_ntree)), output_margin=True
    )


def evaluate_on_valid(
    booster: xgb.Booster,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    best_ntree: int,
) -> dict[str, float | int]:
    observed = build_matrix(valid_df, feature_names)
    risk = predict_xgb_risk(booster, observed, feature_names, best_ntree)
    val_ci = cindex(
        risk,
        valid_df["OS_MONTHS"].to_numpy(float),
        valid_df["OS_STATUS"].to_numpy(int),
    )
    treated = valid_df.copy()
    treated["Adjuvant Chemo"] = 1
    untreated = valid_df.copy()
    untreated["Adjuvant Chemo"] = 0
    risk_treated = predict_xgb_risk(
        booster, build_matrix(treated, feature_names), feature_names, best_ntree
    )
    risk_untreated = predict_xgb_risk(
        booster, build_matrix(untreated, feature_names), feature_names, best_ntree
    )
    recommendation = (risk_treated < risk_untreated).astype(int)
    return {
        "val_ci": val_ci,
        "val_rmst_diff": alignment_rmst_difference(valid_df, recommendation),
        "n_features": len(feature_names),
    }
