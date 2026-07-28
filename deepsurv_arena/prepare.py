"""Frozen clinical-only data, scaling, and validation utilities for DeepSurv."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

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


def build_matrix_from_feature_names(
    df: pd.DataFrame, feature_names: list[str]
) -> np.ndarray:
    return build_matrix(df, feature_names, dtype=np.float32)


def fit_tabular_transform(matrix: np.ndarray) -> dict[str, np.ndarray]:
    median = np.nanmedian(matrix, axis=0).astype(np.float32)
    imputed = np.where(np.isnan(matrix), median, matrix).astype(np.float32)
    mean = imputed.mean(axis=0).astype(np.float32)
    scale = imputed.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return {"median": median, "mean": mean, "scale": scale}


def apply_tabular_transform(
    matrix: np.ndarray, transform: dict[str, np.ndarray]
) -> np.ndarray:
    imputed = np.where(np.isnan(matrix), transform["median"], matrix).astype(
        np.float32
    )
    return ((imputed - transform["mean"]) / transform["scale"]).astype(np.float32)


@torch.no_grad()
def predict_deepsurv_risk(
    model: torch.nn.Module,
    matrix: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    batches = []
    for start in range(0, len(matrix), batch_size):
        tensor = torch.tensor(
            matrix[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        batches.append(
            torch.clamp(model(tensor), -20, 20)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    return np.concatenate(batches).astype(float)


def evaluate_on_valid(
    model: torch.nn.Module,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    transform: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float | int]:
    observed = apply_tabular_transform(
        build_matrix_from_feature_names(valid_df, feature_names), transform
    )
    risk = predict_deepsurv_risk(model, observed, device)
    val_ci = cindex(
        risk,
        valid_df["OS_MONTHS"].to_numpy(float),
        valid_df["OS_STATUS"].to_numpy(int),
    )
    treated = valid_df.copy()
    treated["Adjuvant Chemo"] = 1
    untreated = valid_df.copy()
    untreated["Adjuvant Chemo"] = 0
    treated_matrix = apply_tabular_transform(
        build_matrix_from_feature_names(treated, feature_names), transform
    )
    untreated_matrix = apply_tabular_transform(
        build_matrix_from_feature_names(untreated, feature_names), transform
    )
    recommendation = (
        predict_deepsurv_risk(model, treated_matrix, device)
        < predict_deepsurv_risk(model, untreated_matrix, device)
    ).astype(int)
    return {
        "val_ci": val_ci,
        "val_rmst_diff": alignment_rmst_difference(valid_df, recommendation),
        "n_features": len(feature_names),
    }


def compute_alignment_rmst_diff_deepsurv(
    model: torch.nn.Module,
    df: pd.DataFrame,
    feature_names: list[str],
    transform: dict[str, np.ndarray],
    device: torch.device,
    tau: float = 60,
) -> float:
    treated = df.copy()
    treated["Adjuvant Chemo"] = 1
    untreated = df.copy()
    untreated["Adjuvant Chemo"] = 0
    treated_matrix = apply_tabular_transform(
        build_matrix_from_feature_names(treated, feature_names), transform
    )
    untreated_matrix = apply_tabular_transform(
        build_matrix_from_feature_names(untreated, feature_names), transform
    )
    recommendation = (
        predict_deepsurv_risk(model, treated_matrix, device)
        < predict_deepsurv_risk(model, untreated_matrix, device)
    ).astype(int)
    return alignment_rmst_difference(df, recommendation, tau=tau)
