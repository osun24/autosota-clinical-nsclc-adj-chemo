"""Clinical-covariate-only Random Survival Forest arena."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import random
import time

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import NSGAIISampler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

try:
    from . import prepare
except ImportError:
    import prepare

np.random.seed(42)
random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ARENA_DIR = Path(__file__).resolve().parent
RUNS_DIR = ARENA_DIR / "runs"
DEFAULT_N_TRIALS = int(os.environ.get("RSF_ARENA_N_TRIALS", "10"))
DEFAULT_BOOTSTRAPS = int(os.environ.get("RSF_ARENA_BOOTSTRAPS", "2"))


def _outcome(df: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(
        event=df["OS_STATUS"].astype(bool).to_numpy(),
        time=df["OS_MONTHS"].astype(float).to_numpy(),
    )


def make_rsf(**params) -> RandomSurvivalForest:
    return RandomSurvivalForest(**params)


def _suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 300, 1200, step=100
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 6, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 20),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "max_depth": trial.suggest_categorical(
            "max_depth", [None, 4, 6, 8, 12]
        ),
        "n_jobs": -1,
        "random_state": 42,
    }


def fit_model_from_metadata(
    metadata: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    refit_train_valid: bool = False,
) -> tuple[RandomSurvivalForest, list[str], list[str]]:
    feature_names = list(metadata["feature_names"])
    fit_df = (
        pd.concat([train_df, valid_df], ignore_index=True)
        if refit_train_valid
        else train_df
    )
    weights, _, _ = prepare.compute_iptw(
        fit_df, covariate_cols=list(metadata["pretreatment_columns"])
    )
    model = make_rsf(**dict(metadata["rsf_params"]))
    model.fit(
        prepare.build_matrix_from_feature_names(fit_df, feature_names),
        _outcome(fit_df),
        sample_weight=weights,
    )
    return model, feature_names, list(metadata["clinical_columns"])


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
        key=lambda trial: 0.4 * scaled(trial.values[0], ci)
        + 0.6 * scaled(trial.values[1], rmst),
    )


def _save_artifacts(
    result: dict,
    model: RandomSurvivalForest,
    metadata: dict,
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir()
    with (run_dir / "rsf_model.pkl").open("wb") as stream:
        pickle.dump(model, stream)
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
    feature_names = clinical_columns

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = _suggest_params(trial)
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
            model = make_rsf(**params)
            model.fit(
                prepare.build_matrix_from_feature_names(
                    sample, feature_names
                ),
                _outcome(sample),
                sample_weight=weights,
            )
            scores.append(
                prepare.evaluate_on_valid(model, valid_df, feature_names)
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
        **chosen.params,
        "n_jobs": -1,
        "random_state": 7,
    }
    metadata = {
        "arena": "rsf_clinical",
        "clinical_columns": clinical_columns,
        "pretreatment_columns": pretreatment_columns,
        "feature_names": feature_names,
        "rsf_params": params,
    }
    model, _, _ = fit_model_from_metadata(
        metadata, train_df, valid_df, refit_train_valid=False
    )
    result = {
        **prepare.evaluate_on_valid(model, valid_df, feature_names),
        "chosen_trial": chosen.number,
        "elapsed_seconds": time.time() - started,
    }
    metadata["result"] = result
    if save_artifacts:
        result["run_dir"] = str(_save_artifacts(result, model, metadata))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
