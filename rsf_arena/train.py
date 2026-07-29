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
from sklearn.inspection import permutation_importance
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
DEFAULT_N_TRIALS = int(os.environ.get("RSF_ARENA_N_TRIALS", "30"))
# Bootstraps per trial. iter_003 showed 2 leaves the search ranking configs on
# noise, so the max over trials is largely winner's curse.
DEFAULT_BOOTSTRAPS = int(os.environ.get("RSF_ARENA_BOOTSTRAPS", "8"))

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


def _rsf_params(raw: dict, random_state: int = 42) -> dict:
    """Map a trial's search-space values onto RandomSurvivalForest kwargs.

    `split_multiplier` is a search-space coordinate, not a forest argument, so
    it is resolved here rather than passed through.
    """
    min_samples_leaf = int(raw["min_samples_leaf"])
    return {
        "n_estimators": int(raw["n_estimators"]),
        "min_samples_split": int(
            round(float(raw["split_multiplier"]) * min_samples_leaf)
        ),
        "min_samples_leaf": min_samples_leaf,
        "max_features": float(raw["max_features"]),
        "max_depth": raw["max_depth"],
        "n_jobs": -1,
        "random_state": random_state,
    }


def _suggest_params(trial: optuna.Trial) -> dict:
    # Leaf size drives split size: sampling them independently admits invalid
    # (split < 2*leaf) and redundant combinations. Leaf sizes below 5 are out
    # of scope per the idea library.
    raw = {
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        "split_multiplier": trial.suggest_float("split_multiplier", 2.0, 4.0),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12]),
    }
    return _rsf_params(raw)


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
    w_clip = tuple(metadata.get("iptw_w_clip", DEFAULT_W_CLIP))
    weights, _, _ = prepare.compute_iptw(
        fit_df,
        covariate_cols=list(metadata["pretreatment_columns"]),
        w_clip=w_clip,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed risk, the ACT-vs-OBS recommendation, and its contrast.

    The contrast is `risk(ACT=0) - risk(ACT=1)`, so a positive value favours
    ACT. Averaging contrasts across seeds before thresholding is what the
    ensemble recommendation uses.
    """
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
    contrast = risk_untreated - risk_treated
    return risk, (risk_treated < risk_untreated).astype(int), contrast


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


def _weight_quantiles(weights: np.ndarray) -> dict[str, float]:
    q = np.percentile(weights, [0, 25, 50, 75, 100])
    return {
        "min": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "max": float(q[4]),
    }


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights**2))


def iptw_diagnostics(
    train_df: pd.DataFrame, weights: np.ndarray
) -> dict:
    """ESS and weight quantiles overall and by arm, on training rows only.

    Diagnostic only: read-only summary of weights already fit for training;
    does not change `compute_iptw`, the objective, or the recommendation rule.
    """
    treatment = train_df["Adjuvant Chemo"].to_numpy(int)
    report = {
        "n": len(weights),
        "ess": _effective_sample_size(weights),
        "quantiles": _weight_quantiles(weights),
    }
    for arm, label in ((1, "act"), (0, "obs")):
        mask = treatment == arm
        arm_weights = weights[mask]
        report[label] = {
            "n": int(mask.sum()),
            "ess": _effective_sample_size(arm_weights),
            "quantiles": _weight_quantiles(arm_weights),
        }
    return report


# Weight-clip candidates for iptw_clip_sweep. Selection uses training-only
# ESS, never validation, per the idea's admissibility restriction.
CANDIDATE_W_CLIPS: tuple[tuple[float, float], ...] = (
    (0.1, 5.0),
    (0.1, 10.0),
    (0.1, 20.0),
    (0.05, 10.0),
)
DEFAULT_W_CLIP = (0.1, 10.0)


def select_iptw_clip(
    train_df: pd.DataFrame, pretreatment_columns: list[str]
) -> tuple[tuple[float, float], list[dict]]:
    """Pick a weight-clip range from training-only ESS, never validation."""
    scored = []
    for w_clip in CANDIDATE_W_CLIPS:
        weights, _, _ = prepare.compute_iptw(
            train_df, covariate_cols=pretreatment_columns, w_clip=w_clip
        )
        diag = iptw_diagnostics(train_df, weights)
        scored.append({"w_clip": list(w_clip), **diag})
    best = max(scored, key=lambda row: row["ess"])
    return tuple(best["w_clip"]), scored


def feature_importance_diagnostic(
    model: RandomSurvivalForest,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    n_repeats: int = 20,
    seed: int = 13,
) -> dict:
    """Permutation importance (C-index drop) of the persisted panel model.

    Measures C-index sensitivity only; there is no simple RMST-compatible
    scorer for `permutation_importance`'s API. Diagnostic only: uses the
    already-fit model and validation rows exactly as the frozen evaluator
    does, and does not feed back into fitting, search, or feature selection.
    """
    x_valid = prepare.build_matrix_from_feature_names(valid_df, feature_names)
    y_valid = _outcome(valid_df)
    result = permutation_importance(
        model, x_valid, y_valid, n_repeats=n_repeats, random_state=seed
    )
    ranked = sorted(
        zip(feature_names, result.importances_mean, result.importances_std),
        key=lambda row: row[1],
        reverse=True,
    )
    return {
        "n_repeats": n_repeats,
        "ranked": [
            {
                "feature": name,
                "importance_mean": float(mean),
                "importance_std": float(sd),
            }
            for name, mean, sd in ranked
        ],
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
    w_clip = tuple(metadata.get("iptw_w_clip", DEFAULT_W_CLIP))
    weights, _, _ = prepare.compute_iptw(
        train_df,
        covariate_cols=list(metadata["pretreatment_columns"]),
        w_clip=w_clip,
    )
    matrix = prepare.build_matrix_from_feature_names(train_df, feature_names)
    outcome = _outcome(train_df)
    base_params = {
        key: value
        for key, value in metadata["rsf_params"].items()
        if key != "random_state"
    }

    primary_model = None
    risks, recommendations, contrasts, per_seed = [], [], [], []
    for seed in SEED_PANEL:
        # oob_score is a training-side diagnostic only; it is deliberately kept
        # out of metadata["rsf_params"] so the human finalizer is unaffected.
        model = make_rsf(**base_params, random_state=seed, oob_score=True)
        model.fit(matrix, outcome, sample_weight=weights)
        risk, recommendation, contrast = _valid_predictions(
            model, valid_df, feature_names
        )
        val_ci, val_rmst_diff = _metric_pair(valid_df, risk, recommendation)
        per_seed.append(
            {
                "seed": int(seed),
                "val_ci": val_ci,
                "val_rmst_diff": val_rmst_diff,
                "act_recommended_frac": float(recommendation.mean()),
                "oob_ci": float(getattr(model, "oob_score_", float("nan"))),
            }
        )
        risks.append(risk)
        recommendations.append(recommendation)
        contrasts.append(contrast)
        # Only the primary seed's forest is retained; each one is large.
        if primary_model is None:
            primary_model = model

    risks = np.asarray(risks)
    recommendations = np.asarray(recommendations)
    contrasts = np.asarray(contrasts)
    # Ensemble policy: average the counterfactual contrast across the panel,
    # then threshold. Deterministic given the fixed seed panel.
    ensemble_risk = risks.mean(axis=0)
    ensemble_contrast = contrasts.mean(axis=0)
    ensemble_recommendation = (ensemble_contrast > 0).astype(int)
    panel_ci = np.array([row["val_ci"] for row in per_seed])
    panel_rmst = np.array([row["val_rmst_diff"] for row in per_seed])

    rng = np.random.default_rng(METRIC_DRAW_SEED)
    n_valid = len(valid_df)
    draw_ci, draw_rmst = [], []
    draw_ens_ci, draw_ens_rmst = [], []
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
        ens_ci, ens_rmst = _metric_pair(
            resampled,
            ensemble_risk[index],
            ensemble_recommendation[index],
        )
        draw_ens_ci.append(ens_ci)
        draw_ens_rmst.append(ens_rmst)
    draw_ci = np.asarray(draw_ci)
    draw_rmst = np.asarray(draw_rmst)
    draw_ens_ci = np.asarray(draw_ens_ci)
    draw_ens_rmst = np.asarray(draw_ens_rmst)

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

    def summarize_fixed(point: float, draws: np.ndarray) -> dict[str, float]:
        """Interval for a policy that is deterministic given the panel."""
        draw_spread = _spread(draws)
        return {
            "point": float(point),
            "se_total": draw_spread["sd"],
            "se_boot": draw_spread["sd"],
            "iqr_boot": draw_spread["iqr"],
            "sd_seed": 0.0,
        }

    ensemble_ci, ensemble_rmst = _metric_pair(
        valid_df, ensemble_risk, ensemble_recommendation
    )
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
        "oob_ci": _spread(np.array([row["oob_ci"] for row in per_seed])),
        "iptw_diagnostics": iptw_diagnostics(train_df, weights),
        "optimism_gap_ci": float(
            panel_ci.mean()
            - float(np.mean([row["oob_ci"] for row in per_seed]))
        ),
        "ensemble": {
            "val_ci": summarize_fixed(ensemble_ci, draw_ens_ci),
            "val_rmst_diff": summarize_fixed(ensemble_rmst, draw_ens_rmst),
            "act_recommended_frac": float(ensemble_recommendation.mean()),
            "flipped_vs_primary_seed": int(
                np.sum(ensemble_recommendation != recommendations[0])
            ),
        },
        "per_seed": per_seed,
    }
    return report, primary_model


DEPTH_SWEEP_CANDIDATES: tuple[int | None, ...] = (3, 4, 5, 6, 7, 8, 9, 10, None)
DEPTH_SWEEP_BASE_PARAMS = {
    "n_estimators": 700,
    "min_samples_leaf": 15,
    "min_samples_split": 37,
    "max_features": 0.7,
    "n_jobs": -1,
}


def depth_sweep(
    metadata: dict, train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> list[dict]:
    """Seed-panel depth comparison, holding every other param fixed.

    Read-only diagnostic sweep: does not touch `_suggest_params`, the Optuna
    search space, or which model gets persisted. Isolates depth from the
    search-selection noise documented in iter_003 and iter_009. Skips the
    validation-row bootstrap (irrelevant to comparing depths against a fixed
    validation set) and reports only the panel mean and seed spread.
    """
    feature_names = list(metadata["feature_names"])
    w_clip = tuple(metadata.get("iptw_w_clip", DEFAULT_W_CLIP))
    weights, _, _ = prepare.compute_iptw(
        train_df,
        covariate_cols=list(metadata["pretreatment_columns"]),
        w_clip=w_clip,
    )
    matrix = prepare.build_matrix_from_feature_names(train_df, feature_names)
    outcome = _outcome(train_df)

    rows = []
    for depth in DEPTH_SWEEP_CANDIDATES:
        params = {**DEPTH_SWEEP_BASE_PARAMS, "max_depth": depth}
        panel_ci, panel_rmst = [], []
        for seed in SEED_PANEL:
            model = make_rsf(**params, random_state=seed)
            model.fit(matrix, outcome, sample_weight=weights)
            risk, recommendation, _ = _valid_predictions(
                model, valid_df, feature_names
            )
            val_ci, val_rmst_diff = _metric_pair(valid_df, risk, recommendation)
            panel_ci.append(val_ci)
            panel_rmst.append(val_rmst_diff)
        panel_ci = np.array(panel_ci)
        panel_rmst = np.array(panel_rmst)
        rows.append(
            {
                "max_depth": depth,
                "val_ci": float(panel_ci.mean()),
                "val_ci_sd_seed": float(panel_ci.std(ddof=1)),
                "val_rmst_diff": float(panel_rmst.mean()),
                "val_rmst_diff_sd_seed": float(panel_rmst.std(ddof=1)),
            }
        )
    return rows


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


def _search_diagnostics(
    study: optuna.Study, chosen: optuna.trial.FrozenTrial
) -> dict:
    """How much of the headline is the search picking a lucky trial.

    Diagnostic only: it records what the search already computed and never
    feeds back into selection.
    """
    completed = [t for t in study.trials if t.values is not None]
    ci = np.array([t.values[0] for t in completed])
    rmst = np.array([t.values[1] for t in completed])
    return {
        "n_completed_trials": len(completed),
        "chosen_trial": chosen.number,
        "chosen_values": [float(v) for v in chosen.values],
        # 1 = the search picked the best trial on that objective.
        "chosen_rank_ci": int((ci > chosen.values[0]).sum() + 1),
        "chosen_rank_rmst": int((rmst > chosen.values[1]).sum() + 1),
        "trial_ci": _spread(ci) | {"min": float(ci.min()), "max": float(ci.max())},
        "trial_rmst": _spread(rmst)
        | {"min": float(rmst.min()), "max": float(rmst.max())},
        "pareto_front_size": len(study.best_trials),
    }


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
    # Retain only this run's model and the reigning champion's; the verdict
    # that decides which one survives is recorded in log.md afterwards.
    keep = {run_dir}
    champion = _champion_dir()
    if champion is not None:
        keep.add(champion)
    result["pruned_pickles"] = prune_run_pickles(keep)
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
    # iter_007-009: interaction terms (prepare.INTERACTION_TERMS) were tried
    # and not adopted — the RMST association did not survive trimming to the
    # terms permutation importance called load-bearing (iter_009). Left
    # available in prepare.py at zero cost for a future, better-controlled
    # attempt.
    feature_names = clinical_columns
    chosen_w_clip, clip_candidates = select_iptw_clip(
        train_df, pretreatment_columns
    )

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
                sample,
                covariate_cols=pretreatment_columns,
                w_clip=chosen_w_clip,
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
    params = _rsf_params(chosen.params, random_state=7)
    metadata = {
        "arena": "rsf_clinical",
        "clinical_columns": clinical_columns,
        "pretreatment_columns": pretreatment_columns,
        "feature_names": feature_names,
        "rsf_params": params,
        "iptw_w_clip": list(chosen_w_clip),
    }
    panel, model = seed_panel_report(metadata, train_df, valid_df)
    result = {
        **prepare.evaluate_on_valid(model, valid_df, feature_names),
        "chosen_trial": chosen.number,
        "search": _search_diagnostics(study, chosen),
        "iptw_clip_candidates": clip_candidates,
        "seed_panel": panel,
        "feature_importance": feature_importance_diagnostic(
            model, valid_df, feature_names
        ),
        "depth_sweep": depth_sweep(metadata, train_df, valid_df),
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
