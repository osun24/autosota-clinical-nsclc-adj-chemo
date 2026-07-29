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
# Names the run directory whose forest pickle is worth keeping on disk.
BEST_POINTER = RUNS_DIR / "best_run.txt"
DEFAULT_N_TRIALS = int(os.environ.get("RSF_ARENA_N_TRIALS", "10"))
DEFAULT_BOOTSTRAPS = int(os.environ.get("RSF_ARENA_BOOTSTRAPS", "2"))

# Prespecified forest seeds for the reporting panel. The first entry is the
# seed the persisted model uses, so artifacts keep their previous meaning.
SEED_PANEL = (7, 17, 27, 37, 47, 57, 67, 77, 87, 97)
# Validation-row resamples used to put an interval around an already-computed
# metric. This never touches fitting or hyperparameter selection.
DEFAULT_METRIC_DRAWS = int(os.environ.get("RSF_ARENA_METRIC_DRAWS", "500"))
METRIC_DRAW_SEED = 20260728


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


def _valid_predictions(
    model: RandomSurvivalForest,
    valid_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Observed risk and the counterfactual ACT-vs-OBS recommendation."""
    risk = prepare.predict_rsf_risk(
        model, prepare.build_matrix_from_feature_names(valid_df, feature_names)
    )
    treated = valid_df.copy()
    treated["Adjuvant Chemo"] = 1
    untreated = valid_df.copy()
    untreated["Adjuvant Chemo"] = 0
    risk_treated = prepare.predict_rsf_risk(
        model, prepare.build_matrix_from_feature_names(treated, feature_names)
    )
    risk_untreated = prepare.predict_rsf_risk(
        model, prepare.build_matrix_from_feature_names(untreated, feature_names)
    )
    return risk, (risk_treated < risk_untreated).astype(int)


def _metric_pair(
    valid_df: pd.DataFrame, risk: np.ndarray, recommendation: np.ndarray
) -> tuple[float, float]:
    """Both objectives, using the frozen metric functions unchanged."""
    return (
        prepare.cindex(
            risk,
            valid_df["OS_MONTHS"].to_numpy(float),
            valid_df["OS_STATUS"].to_numpy(int),
        ),
        prepare.alignment_rmst_difference(valid_df, recommendation),
    )


def _spread(values: np.ndarray) -> dict[str, float]:
    quartiles = np.percentile(values, [25, 75])
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "iqr": float(quartiles[1] - quartiles[0]),
    }


def seed_panel_report(
    metadata: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    metric_draws: int = DEFAULT_METRIC_DRAWS,
) -> tuple[dict, RandomSurvivalForest]:
    """Refit across a fixed seed panel and quantify both objectives.

    Uncertainty has two components: spread across forest seeds, and the
    sampling spread of the validation rows. The panel refits on training rows
    only; the row resampling is a post-hoc interval on an already-computed
    metric and never informs fitting or model selection.
    """
    feature_names = list(metadata["feature_names"])
    weights, _, _ = prepare.compute_iptw(
        train_df, covariate_cols=list(metadata["pretreatment_columns"])
    )
    matrix = prepare.build_matrix_from_feature_names(train_df, feature_names)
    outcome = _outcome(train_df)
    base_params = {
        key: value
        for key, value in metadata["rsf_params"].items()
        if key != "random_state"
    }

    primary_model = None
    risks, recommendations, per_seed = [], [], []
    for seed in SEED_PANEL:
        model = make_rsf(**base_params, random_state=seed)
        model.fit(matrix, outcome, sample_weight=weights)
        risk, recommendation = _valid_predictions(
            model, valid_df, feature_names
        )
        val_ci, val_rmst_diff = _metric_pair(valid_df, risk, recommendation)
        per_seed.append(
            {
                "seed": int(seed),
                "val_ci": val_ci,
                "val_rmst_diff": val_rmst_diff,
                "act_recommended_frac": float(recommendation.mean()),
            }
        )
        risks.append(risk)
        recommendations.append(recommendation)
        # Only the primary seed's forest is retained; each one is large.
        if primary_model is None:
            primary_model = model

    risks = np.asarray(risks)
    recommendations = np.asarray(recommendations)
    panel_ci = np.array([row["val_ci"] for row in per_seed])
    panel_rmst = np.array([row["val_rmst_diff"] for row in per_seed])

    rng = np.random.default_rng(METRIC_DRAW_SEED)
    n_valid = len(valid_df)
    draw_ci, draw_rmst = [], []
    for _ in range(metric_draws):
        index = rng.integers(0, n_valid, size=n_valid)
        resampled = valid_df.iloc[index]
        if int(resampled["OS_STATUS"].sum()) == 0:
            continue
        pairs = [
            _metric_pair(resampled, risks[k][index], recommendations[k][index])
            for k in range(len(SEED_PANEL))
        ]
        draw_ci.append(float(np.mean([pair[0] for pair in pairs])))
        draw_rmst.append(float(np.mean([pair[1] for pair in pairs])))
    draw_ci = np.asarray(draw_ci)
    draw_rmst = np.asarray(draw_rmst)

    def summarize(
        panel: np.ndarray, draws: np.ndarray
    ) -> dict[str, float]:
        seed_spread = _spread(panel)
        draw_spread = _spread(draws)
        se_seed = seed_spread["sd"] / np.sqrt(len(panel))
        return {
            "point": seed_spread["mean"],
            "se_total": float(
                np.sqrt(draw_spread["sd"] ** 2 + se_seed**2)
            ),
            "se_boot": draw_spread["sd"],
            "iqr_boot": draw_spread["iqr"],
            "sd_seed": seed_spread["sd"],
        }

    act_fraction = recommendations.mean(axis=0)
    modal_fraction = np.maximum(act_fraction, 1.0 - act_fraction)
    report = {
        "n_seeds": len(SEED_PANEL),
        "seeds": [int(seed) for seed in SEED_PANEL],
        "metric_draws": int(len(draw_ci)),
        "val_ci": summarize(panel_ci, draw_ci),
        "val_rmst_diff": summarize(panel_rmst, draw_rmst),
        "recommendation_agreement": float(modal_fraction.mean()),
        "recommendation_unanimous_frac": float(
            np.mean(modal_fraction == 1.0)
        ),
        "act_recommended_frac_mean": float(act_fraction.mean()),
        "per_seed": per_seed,
    }
    return report, primary_model


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


def _champion_dir() -> Path | None:
    """Run directory currently holding the best-iteration model, if any."""
    if not BEST_POINTER.exists():
        return None
    name = BEST_POINTER.read_text().strip()
    candidate = RUNS_DIR / name
    return candidate if candidate.is_dir() else None


def prune_run_pickles(keep: set[Path]) -> list[str]:
    """Delete every forest pickle except the ones in `keep`.

    Each pickle is roughly 350 MB, so only the best iteration's model is worth
    retaining. JSON metadata and results are small and are always kept, so the
    provenance of every run survives pruning.
    """
    resolved = {path.resolve() for path in keep}
    removed = []
    for pickle_path in sorted(RUNS_DIR.glob("run_*/rsf_model.pkl")):
        if pickle_path.parent.resolve() not in resolved:
            pickle_path.unlink()
            removed.append(pickle_path.parent.name)
    return removed


def promote_best(run_dir: str | Path) -> list[str]:
    """Mark a run as the best iteration and drop every other pickle."""
    run_path = RUNS_DIR / Path(run_dir).name
    if not (run_path / "rsf_model.pkl").exists():
        raise FileNotFoundError(f"No retained model pickle in {run_path}")
    BEST_POINTER.write_text(run_path.name + "\n")
    return prune_run_pickles({run_path})


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
    # Retain only this run's model and the reigning champion's; the verdict
    # that decides which one survives is recorded in log.md afterwards.
    keep = {run_dir}
    champion = _champion_dir()
    if champion is not None:
        keep.add(champion)
    result["pruned_pickles"] = prune_run_pickles(keep)
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
    panel, model = seed_panel_report(metadata, train_df, valid_df)
    result = {
        **prepare.evaluate_on_valid(model, valid_df, feature_names),
        "chosen_trial": chosen.number,
        "seed_panel": panel,
        "elapsed_seconds": time.time() - started,
    }
    # The panel's first seed is the persisted model, so its metrics must match
    # the frozen evaluator exactly. Guards against the panel drifting from it.
    primary = panel["per_seed"][0]
    for key in ("val_ci", "val_rmst_diff"):
        if not np.isclose(primary[key], result[key], rtol=0, atol=1e-12):
            raise RuntimeError(
                f"Seed-panel {key} disagrees with the frozen evaluator: "
                f"{primary[key]!r} vs {result[key]!r}"
            )
    metadata["result"] = result
    if save_artifacts:
        result["run_dir"] = str(_save_artifacts(result, model, metadata))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
