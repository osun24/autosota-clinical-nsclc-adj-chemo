"""Frozen clinical-only data and validation utilities for RSF."""

from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

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

# Prespecified, fixed-budget interaction terms (clinical_interaction_expansion).
# Each is a product of two main-effect columns that remain in the feature set,
# so the expansion is hierarchical rather than replacing any main effect.
INTERACTION_TERMS = (
    "ACT_x_Age",
    "ACT_x_StageIII",
    "ACT_x_Adenocarcinoma",
    "ACT_x_Male",
    "Age_x_StageIII",
    "Age_x_Smoked_Yes",
)


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the fixed interaction terms from base columns.

    Always derived from the current `Adjuvant Chemo` value, so counterfactual
    ACT flips (treated/untreated copies) propagate correctly into the
    ACT-interaction terms instead of carrying stale values forward.
    """
    out = df.copy()
    act = out["Adjuvant Chemo"].astype(float)
    age = out["Age"].astype(float)
    out["ACT_x_Age"] = act * age
    out["ACT_x_StageIII"] = act * out["Stage_III"].astype(float)
    out["ACT_x_Adenocarcinoma"] = act * out["Histology_Adenocarcinoma"].astype(
        float
    )
    out["ACT_x_Male"] = act * out["IS_MALE"].astype(float)
    out["Age_x_StageIII"] = age * out["Stage_III"].astype(float)
    out["Age_x_Smoked_Yes"] = age * out["Smoked?_Yes"].astype(float)
    return out


def build_matrix_from_feature_names(
    df: pd.DataFrame, feature_names: list[str]
) -> np.ndarray:
    enriched = add_interaction_terms(df)
    return build_matrix(enriched, feature_names, dtype=np.float64)


def predict_rsf_risk(model, matrix: np.ndarray) -> np.ndarray:
    with threadpool_limits(limits=None):
        return np.asarray(model.predict(matrix.astype(np.float64)), dtype=float)


def evaluate_on_valid(
    model, valid_df: pd.DataFrame, feature_names: list[str]
) -> dict[str, float | int]:
    risk = predict_rsf_risk(
        model, build_matrix_from_feature_names(valid_df, feature_names)
    )
    val_ci = cindex(
        risk,
        valid_df["OS_MONTHS"].to_numpy(float),
        valid_df["OS_STATUS"].to_numpy(int),
    )
    treated = valid_df.copy()
    treated["Adjuvant Chemo"] = 1
    untreated = valid_df.copy()
    untreated["Adjuvant Chemo"] = 0
    risk_treated = predict_rsf_risk(
        model, build_matrix_from_feature_names(treated, feature_names)
    )
    risk_untreated = predict_rsf_risk(
        model, build_matrix_from_feature_names(untreated, feature_names)
    )
    recommendation = (risk_treated < risk_untreated).astype(int)
    return {
        "val_ci": val_ci,
        "val_rmst_diff": alignment_rmst_difference(valid_df, recommendation),
        "n_features": len(feature_names),
    }
