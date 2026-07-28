"""Clinical-covariate-only XGBoost-Cox T-learner arena."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from optuna.samplers import NSGAIISampler

try:
    from . import prepare
except ImportError:
    import prepare

np.random.seed(42)
random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ARENA_DIR = Path(__file__).resolve().parent
RUNS_DIR = ARENA_DIR / "runs"
DEFAULT_N_TRIALS = int(os.environ.get("XGB_ARENA_N_TRIALS", "10"))
DEFAULT_BOOTSTRAPS = int(os.environ.get("XGB_ARENA_BOOTSTRAPS", "2"))


def make_dmatrix(
    matrix: np.ndarray,
    time_vals: np.ndarray,
    events: np.ndarray,
    weight: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> xgb.DMatrix:
    return xgb.DMatrix(
        matrix.astype(np.float32),
        label=prepare.pack_cox_labels(time_vals, events),
        weight=weight,
        feature_names=feature_names,
    )


def xgb_cindex_eval(
    predictions: np.ndarray, data: xgb.DMatrix
) -> tuple[str, float]:
    labels = data.get_label()
    return "cindex", prepare.cindex(
        predictions, np.abs(labels), (labels > 0).astype(int)
    )


def train_xgb_cox(
    dtrain: xgb.DMatrix,
    dvalid: xgb.DMatrix,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[xgb.Booster, dict]:
    history: dict = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (dvalid, "valid")],
        custom_metric=xgb_cindex_eval,
        evals_result=history,
        early_stopping_rounds=int(early_stopping_rounds),
        maximize=True,
        verbose_eval=False,
    )
    return booster, history


def _suggest_params(trial: optuna.Trial) -> tuple[dict, int, int]:
    params = {
        "objective": "survival:cox",
        "booster": "gbtree",
        "tree_method": "hist",
        "disable_default_eval_metric": True,
        "eta": trial.suggest_float("eta", 0.01, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 2.0, 30.0, log=True
        ),
        "gamma": trial.suggest_float("gamma", 0.0, 6.0),
        "subsample": trial.suggest_float("subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 3.0, 60.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "seed": 42,
        "nthread": -1,
    }
    rounds = trial.suggest_int("num_boost_round", 300, 1600, step=100)
    stopping = trial.suggest_int("early_stopping_rounds", 50, 150, step=25)
    return params, rounds, stopping


def _fit_arm_pair(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    weights: np.ndarray,
    params: dict,
    rounds: int,
    stopping: int,
) -> tuple[xgb.Booster, xgb.Booster, int, int]:
    models: list[xgb.Booster] = []
    limits: list[int] = []
    for arm in (0, 1):
        train_mask = train_df["Adjuvant Chemo"].to_numpy(int) == arm
        valid_mask = valid_df["Adjuvant Chemo"].to_numpy(int) == arm
        train_arm = train_df.loc[train_mask]
        valid_arm = valid_df.loc[valid_mask]
        dtrain = make_dmatrix(
            train_arm[feature_names].to_numpy(np.float32),
            train_arm["OS_MONTHS"].to_numpy(float),
            train_arm["OS_STATUS"].to_numpy(int),
            weights[train_mask],
            feature_names,
        )
        dvalid = make_dmatrix(
            valid_arm[feature_names].to_numpy(np.float32),
            valid_arm["OS_MONTHS"].to_numpy(float),
            valid_arm["OS_STATUS"].to_numpy(int),
            None,
            feature_names,
        )
        arm_params = dict(params)
        arm_params["seed"] = int(params.get("seed", 42)) + arm
        booster, _ = train_xgb_cox(
            dtrain, dvalid, arm_params, rounds, stopping
        )
        best = (
            booster.best_iteration + 1
            if booster.best_iteration is not None
            else rounds
        )
        models.append(prepare.slice_booster_to_best_iteration(booster, best))
        limits.append(int(best))
    return models[0], models[1], limits[0], limits[1]


def _evaluate_pair(
    model0: xgb.Booster,
    model1: xgb.Booster,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    limit0: int,
    limit1: int,
) -> dict[str, float | int]:
    matrix = valid_df[feature_names].to_numpy(np.float32)
    data = xgb.DMatrix(matrix, feature_names=feature_names)
    risk0 = model0.predict(
        data, iteration_range=(0, limit0), output_margin=True
    )
    risk1 = model1.predict(
        data, iteration_range=(0, limit1), output_margin=True
    )
    recommendation = (risk1 < risk0).astype(int)
    return {
        "val_ci": prepare.cindex(
            risk0,
            valid_df["OS_MONTHS"].to_numpy(float),
            valid_df["OS_STATUS"].to_numpy(int),
        ),
        "val_rmst_diff": prepare.alignment_rmst_difference(
            valid_df, recommendation
        ),
        "n_features": len(feature_names),
    }


def _select_compromise(study: optuna.Study) -> optuna.trial.FrozenTrial:
    candidates = study.best_trials or [
        trial for trial in study.trials if trial.values is not None
    ]
    if not candidates:
        raise RuntimeError("No completed trials.")
    ci = np.array([trial.values[0] for trial in candidates], dtype=float)
    rmst = np.array([trial.values[1] for trial in candidates], dtype=float)

    def scaled(value: float, values: np.ndarray) -> float:
        span = float(values.max() - values.min())
        return 0.0 if span == 0 else (value - float(values.min())) / span

    return max(
        candidates,
        key=lambda trial: scaled(trial.values[0], ci)
        + scaled(trial.values[1], rmst),
    )


def fit_model_from_metadata(
    metadata: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    refit_train_valid: bool = False,
) -> tuple[xgb.Booster, xgb.Booster, int, int, list[str]]:
    feature_names = list(metadata["feature_names"])
    params = dict(metadata["xgb_params"])
    rounds = int(metadata["num_boost_round"])
    stopping = int(metadata["early_stopping_rounds"])
    if refit_train_valid:
        combined = pd.concat([train_df, valid_df], ignore_index=True)
        rng = np.random.default_rng(42)
        valid_indices = []
        for arm in (0, 1):
            indices = np.flatnonzero(
                combined["Adjuvant Chemo"].to_numpy(int) == arm
            )
            valid_indices.extend(
                rng.choice(indices, size=max(2, len(indices) // 4), replace=False)
            )
        valid_mask = np.zeros(len(combined), dtype=bool)
        valid_mask[valid_indices] = True
        fit_df = combined.loc[~valid_mask].reset_index(drop=True)
        stop_df = combined.loc[valid_mask].reset_index(drop=True)
    else:
        fit_df, stop_df = train_df, valid_df
    weights, _, _ = prepare.compute_iptw(
        fit_df, covariate_cols=list(metadata["pretreatment_columns"])
    )
    model0, model1, limit0, limit1 = _fit_arm_pair(
        fit_df,
        stop_df,
        feature_names,
        weights,
        params,
        rounds,
        stopping,
    )
    return model0, model1, limit0, limit1, feature_names


def _save_artifacts(
    result: dict,
    model0: xgb.Booster,
    model1: xgb.Booster,
    metadata: dict,
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir()
    model0.save_model(run_dir / "xgb_model_arm0.json")
    model1.save_model(run_dir / "xgb_model_arm1.json")
    (run_dir / "feature_names.txt").write_text(
        "\n".join(metadata["feature_names"]) + "\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return run_dir


def run(
    n_trials: int = DEFAULT_N_TRIALS,
    bootstrap_n: int = DEFAULT_BOOTSTRAPS,
    save_artifacts: bool = True,
) -> dict:
    started = time.time()
    train_df, valid_df = prepare.load_train_valid()
    clinical_columns, pretreatment_columns = prepare.clinical_columns(
        train_df, valid_df
    )
    feature_names = pretreatment_columns

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params, rounds, stopping = _suggest_params(trial)
        scores = []
        for iteration in range(bootstrap_n):
            sample = prepare.bootstrap_resample_df(
                train_df,
                seed=42 + 1000 * trial.number + iteration,
                require_two_arms=True,
            )
            weights, _, _ = prepare.compute_iptw(
                sample, covariate_cols=pretreatment_columns
            )
            fitted = _fit_arm_pair(
                sample,
                valid_df,
                feature_names,
                weights,
                params,
                rounds,
                stopping,
            )
            scores.append(
                _evaluate_pair(
                    fitted[0],
                    fitted[1],
                    valid_df,
                    feature_names,
                    fitted[2],
                    fitted[3],
                )
            )
        trial.set_user_attr(
            "best_ntree_arm0",
            int(np.median([score.get("best_ntree_arm0", rounds) for score in scores])),
        )
        return (
            float(np.mean([score["val_ci"] for score in scores])),
            float(np.mean([score["val_rmst_diff"] for score in scores])),
        )

    study = optuna.create_study(
        directions=["maximize", "maximize"], sampler=NSGAIISampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)
    chosen = _select_compromise(study)
    params = {
        "objective": "survival:cox",
        "booster": "gbtree",
        "tree_method": "hist",
        "disable_default_eval_metric": True,
        "seed": 7,
        "nthread": -1,
        **{
            key: value
            for key, value in chosen.params.items()
            if key
            not in {"num_boost_round", "early_stopping_rounds"}
        },
    }
    metadata = {
        "arena": "xgb_clinical_tlearner",
        "clinical_columns": clinical_columns,
        "pretreatment_columns": pretreatment_columns,
        "feature_names": feature_names,
        "xgb_params": params,
        "num_boost_round": int(chosen.params["num_boost_round"]),
        "early_stopping_rounds": int(chosen.params["early_stopping_rounds"]),
    }
    model0, model1, limit0, limit1, _ = fit_model_from_metadata(
        metadata, train_df, valid_df
    )
    score = _evaluate_pair(
        model0, model1, valid_df, feature_names, limit0, limit1
    )
    result = {
        **score,
        "best_ntree_arm0": limit0,
        "best_ntree_arm1": limit1,
        "chosen_trial": chosen.number,
        "elapsed_seconds": time.time() - started,
    }
    metadata["result"] = result
    if save_artifacts:
        result["run_dir"] = str(
            _save_artifacts(result, model0, model1, metadata)
        )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
