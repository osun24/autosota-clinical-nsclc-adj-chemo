"""Clinical-covariate-only DeepSurv arena."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import NSGAIISampler
from torch import nn

try:
    from . import prepare
except ImportError:
    import prepare

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ARENA_DIR = Path(__file__).resolve().parent
RUNS_DIR = ARENA_DIR / "runs"
DEFAULT_N_TRIALS = int(os.environ.get("DEEPSURV_ARENA_N_TRIALS", "10"))
DEFAULT_BOOTSTRAPS = int(os.environ.get("DEEPSURV_ARENA_BOOTSTRAPS", "2"))
DEFAULT_MAX_EPOCHS = int(os.environ.get("DEEPSURV_ARENA_EPOCHS", "120"))


class DeepSurvMLP(nn.Module):
    def __init__(
        self, in_features: int, hidden_layers: list[int], dropout: float = 0.0
    ):
        super().__init__()
        layers: list[nn.Module] = []
        width = in_features
        for hidden in hidden_layers:
            layers.extend(
                [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            )
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).reshape(-1)


def cox_breslow_weighted(
    prediction: torch.Tensor,
    event: torch.Tensor,
    time_vals: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    order = torch.argsort(time_vals, descending=True)
    prediction = prediction[order]
    event = event[order]
    weight = weight[order]
    log_risk = torch.logcumsumexp(prediction, dim=0)
    contributions = weight * event * (prediction - log_risk)
    return -contributions.sum() / torch.clamp((weight * event).sum(), min=1.0)


def train_deepsurv_model(
    train_matrix: np.ndarray,
    train_df: pd.DataFrame,
    weights: np.ndarray,
    params: dict,
    device: torch.device,
    max_epochs: int,
) -> DeepSurvMLP:
    torch.manual_seed(int(params.get("seed", 42)))
    model = DeepSurvMLP(
        train_matrix.shape[1],
        list(params["hidden_layers"]),
        float(params["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    matrix = torch.tensor(train_matrix, dtype=torch.float32, device=device)
    event = torch.tensor(
        train_df["OS_STATUS"].to_numpy(float),
        dtype=torch.float32,
        device=device,
    )
    duration = torch.tensor(
        train_df["OS_MONTHS"].to_numpy(float),
        dtype=torch.float32,
        device=device,
    )
    weight = torch.tensor(weights, dtype=torch.float32, device=device)
    model.train()
    for _ in range(int(max_epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = cox_breslow_weighted(model(matrix), event, duration, weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    model.eval()
    return model


def _suggest_params(trial: optuna.Trial) -> dict:
    layout = trial.suggest_categorical(
        "hidden_layout", ["16", "32", "32-16", "64-32"]
    )
    return {
        "hidden_layers": [int(width) for width in layout.split("-")],
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 5e-3, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        ),
        "seed": 42,
    }


def fit_model_from_metadata(
    metadata: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    refit_train_valid: bool = False,
    device: torch.device | None = None,
) -> tuple[
    DeepSurvMLP, dict[str, np.ndarray], list[str], list[str]
]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_names = list(metadata["feature_names"])
    fit_df = (
        pd.concat([train_df, valid_df], ignore_index=True)
        if refit_train_valid
        else train_df
    )
    raw = prepare.build_matrix_from_feature_names(fit_df, feature_names)
    transform = prepare.fit_tabular_transform(raw)
    matrix = prepare.apply_tabular_transform(raw, transform)
    weights, _, _ = prepare.compute_iptw(
        fit_df, covariate_cols=list(metadata["pretreatment_columns"])
    )
    model = train_deepsurv_model(
        matrix,
        fit_df,
        weights,
        dict(metadata["deepsurv_params"]),
        device,
        int(metadata["max_epochs"]),
    )
    return model, transform, feature_names, list(metadata["clinical_columns"])


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


def _save_artifacts(
    result: dict,
    model: DeepSurvMLP,
    metadata: dict,
    transform: dict[str, np.ndarray],
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir()
    torch.save(model.state_dict(), run_dir / "deepsurv_model.pt")
    np.savez(run_dir / "tabular_transform.npz", **transform)
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
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    save_artifacts: bool = True,
) -> dict:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            raw = prepare.build_matrix_from_feature_names(
                sample, feature_names
            )
            transform = prepare.fit_tabular_transform(raw)
            matrix = prepare.apply_tabular_transform(raw, transform)
            weights, _, _ = prepare.compute_iptw(
                sample, covariate_cols=pretreatment_columns
            )
            model = train_deepsurv_model(
                matrix, sample, weights, params, device, max_epochs
            )
            scores.append(
                prepare.evaluate_on_valid(
                    model, valid_df, feature_names, transform, device
                )
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
        "hidden_layers": [
            int(width) for width in chosen.params["hidden_layout"].split("-")
        ],
        "dropout": chosen.params["dropout"],
        "learning_rate": chosen.params["learning_rate"],
        "weight_decay": chosen.params["weight_decay"],
        "seed": 7,
    }
    metadata = {
        "arena": "deepsurv_clinical",
        "clinical_columns": clinical_columns,
        "pretreatment_columns": pretreatment_columns,
        "feature_names": feature_names,
        "deepsurv_params": params,
        "max_epochs": int(max_epochs),
    }
    model, transform, _, _ = fit_model_from_metadata(
        metadata, train_df, valid_df, device=device
    )
    result = {
        **prepare.evaluate_on_valid(
            model, valid_df, feature_names, transform, device
        ),
        "chosen_trial": chosen.number,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
    }
    metadata["result"] = result
    if save_artifacts:
        result["run_dir"] = str(
            _save_artifacts(result, model, metadata, transform)
        )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
