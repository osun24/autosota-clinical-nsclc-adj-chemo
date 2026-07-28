"""Shared clinical-only data loading and survival-analysis utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from sklearn.linear_model import LogisticRegression
from sksurv.metrics import concordance_index_censored

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = REPO_ROOT / "clinicalTrain.csv"
VALID_CSV = REPO_ROOT / "clinicalValidation.csv"

SURVIVAL_COLUMNS = ["OS_STATUS", "OS_MONTHS"]
CLINICAL_VARS = [
    "Adjuvant Chemo",
    "Age",
    "IS_MALE",
    "Stage_IA",
    "Stage_IB",
    "Stage_II",
    "Stage_III",
    "Histology_Adenocarcinoma",
    "Histology_Adenosquamous Carcinoma",
    "Histology_Large Cell Carcinoma",
    "Histology_Squamous Cell Carcinoma",
    "Race_African American",
    "Race_Asian",
    "Race_Caucasian",
    "Race_Native Hawaiian or Other Pacific Islander",
    "Race_Unknown",
    "Smoked?_No",
    "Smoked?_Unknown",
    "Smoked?_Yes",
]
PRETREATMENT_VARS = [c for c in CLINICAL_VARS if c != "Adjuvant Chemo"]
ANALYSIS_COLUMNS = SURVIVAL_COLUMNS + CLINICAL_VARS
BOOTSTRAP_MAX_RESAMPLE_TRIES = 256


def assert_not_test_path(path: str | Path) -> None:
    assert "test" not in str(path).lower(), f"Test path is sealed during loop: {path}"


def preprocess_split(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ANALYSIS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required clinical columns: {missing}")

    out = df[ANALYSIS_COLUMNS].copy()
    mapped = out["Adjuvant Chemo"].map({"OBS": 0, "ACT": 1})
    out["Adjuvant Chemo"] = mapped.where(mapped.notna(), out["Adjuvant Chemo"])
    for column in ["Adjuvant Chemo", "IS_MALE", "OS_STATUS"]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
    out["OS_MONTHS"] = pd.to_numeric(out["OS_MONTHS"], errors="raise").astype(float)
    for column in CLINICAL_VARS:
        out[column] = pd.to_numeric(out[column], errors="raise")
    return out


def load_train_valid(
    train_csv: str | Path = TRAIN_CSV,
    valid_csv: str | Path = VALID_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_not_test_path(train_csv)
    assert_not_test_path(valid_csv)
    train_df = preprocess_split(pd.read_csv(train_csv))
    valid_df = preprocess_split(pd.read_csv(valid_csv))
    sort_cols = ["OS_MONTHS", "OS_STATUS"]
    train_df = train_df.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)
    valid_df = valid_df.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)
    return train_df, valid_df


def clinical_columns(
    train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    shared = set(train_df.columns) & set(valid_df.columns)
    all_columns = [column for column in CLINICAL_VARS if column in shared]
    pretreatment = [column for column in all_columns if column != "Adjuvant Chemo"]
    return all_columns, pretreatment


def compute_iptw(
    df: pd.DataFrame,
    covariate_cols: list[str],
    act_col: str = "Adjuvant Chemo",
    ps_clip: tuple[float, float] = (0.05, 0.95),
    w_clip: tuple[float, float] = (0.1, 10.0),
    ref_prev: float | None = None,
    model: LogisticRegression | None = None,
) -> tuple[np.ndarray, LogisticRegression, float]:
    treatment = df[act_col].astype(int).to_numpy()
    covariates = df[covariate_cols].astype(float).to_numpy()
    if model is None:
        model = LogisticRegression(
            max_iter=2000, solver="lbfgs", class_weight="balanced"
        )
        model.fit(covariates, treatment)
    propensity = np.clip(
        model.predict_proba(covariates)[:, 1], ps_clip[0], ps_clip[1]
    )
    if ref_prev is None:
        ref_prev = float(treatment.mean())
    weights = np.where(
        treatment == 1,
        ref_prev / propensity,
        (1 - ref_prev) / (1 - propensity),
    )
    return np.clip(weights, *w_clip), model, float(ref_prev)


def cindex(pred: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            event.astype(bool), time.astype(float), pred.astype(float)
        )[0]
    )


def build_matrix(df: pd.DataFrame, feature_names: list[str], dtype=np.float32) -> np.ndarray:
    missing = [column for column in feature_names if column not in df.columns]
    if missing:
        raise ValueError(f"Data is missing model features: {missing}")
    return df[feature_names].to_numpy(dtype=dtype)


def alignment_rmst_difference(
    df: pd.DataFrame, recommendation: np.ndarray, tau: float = 60
) -> float:
    alignment = df["Adjuvant Chemo"].to_numpy(dtype=int) == recommendation.astype(int)
    if int(alignment.sum()) == 0 or int((~alignment).sum()) == 0:
        return 0.0
    aligned = KaplanMeierFitter().fit(
        df.loc[alignment, "OS_MONTHS"],
        event_observed=df.loc[alignment, "OS_STATUS"],
    )
    unaligned = KaplanMeierFitter().fit(
        df.loc[~alignment, "OS_MONTHS"],
        event_observed=df.loc[~alignment, "OS_STATUS"],
    )
    return float(
        restricted_mean_survival_time(aligned, t=tau)
        - restricted_mean_survival_time(unaligned, t=tau)
    )


def bootstrap_resample_df(
    df: pd.DataFrame,
    seed: int,
    require_two_arms: bool = False,
    require_event: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    for _ in range(BOOTSTRAP_MAX_RESAMPLE_TRIES):
        sample = df.iloc[rng.integers(0, len(df), size=len(df))].copy()
        if require_event and int(sample["OS_STATUS"].sum()) == 0:
            continue
        if require_two_arms and sample["Adjuvant Chemo"].nunique() < 2:
            continue
        return sample.sort_values(
            ["OS_MONTHS", "OS_STATUS"], ascending=[False, False]
        ).reset_index(drop=True)
    raise RuntimeError("Could not draw a bootstrap sample satisfying the constraints.")
