"""Human-only sealed-test evaluation for clinical XGBoost artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from clinical_data import alignment_rmst_difference
from xgb_arena import prepare
from xgb_arena import train

REPO_ROOT = Path(__file__).resolve().parent
TEST_CSV = REPO_ROOT / "affyfRMATest.csv"
XGB_RUNS_DIR = REPO_ROOT / "xgb_arena" / "runs"


def _latest_run_dir(runs_dir: Path = XGB_RUNS_DIR) -> Path:
    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "metadata.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run found in {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_test() -> pd.DataFrame:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"{TEST_CSV} does not exist. Add the sealed test CSV and rerun manually."
        )
    return prepare.preprocess_split(pd.read_csv(TEST_CSV)).sort_values(
        ["OS_MONTHS", "OS_STATUS"], ascending=[False, False]
    ).reset_index(drop=True)


def evaluate_xgb(
    run_dir: Path,
    refit_train_valid: bool = True,
    km_figure_path: Path | None = None,
) -> dict:
    del km_figure_path
    run_dir = run_dir.resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    feature_names = list(metadata["feature_names"])
    test_df = _load_test()
    if refit_train_valid:
        train_df, valid_df = prepare.load_train_valid()
        model0, model1, limit0, limit1, feature_names = (
            train.fit_model_from_metadata(
                metadata, train_df, valid_df, refit_train_valid=True
            )
        )
        source = "refit_train_valid"
    else:
        model0 = xgb.Booster()
        model1 = xgb.Booster()
        model0.load_model(run_dir / "xgb_model_arm0.json")
        model1.load_model(run_dir / "xgb_model_arm1.json")
        limit0 = int(metadata["result"]["best_ntree_arm0"])
        limit1 = int(metadata["result"]["best_ntree_arm1"])
        source = "saved_validation_model"

    matrix = test_df[feature_names].to_numpy(np.float32)
    data = xgb.DMatrix(matrix, feature_names=feature_names)
    risk0 = model0.predict(
        data, iteration_range=(0, limit0), output_margin=True
    )
    risk1 = model1.predict(
        data, iteration_range=(0, limit1), output_margin=True
    )
    recommendation = (risk1 < risk0).astype(int)
    return {
        "model": "xgb_clinical_tlearner",
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "model_source": source,
        "test_ci": prepare.cindex(
            risk0,
            test_df["OS_MONTHS"].to_numpy(float),
            test_df["OS_STATUS"].to_numpy(int),
        ),
        "test_rmst_diff": alignment_rmst_difference(
            test_df, recommendation
        ),
        "n_features": len(feature_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-only sealed-test evaluation for XGBoost."
    )
    parser.add_argument("--xgb-run-dir", type=Path)
    parser.add_argument("--no-refit-train-valid", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "final_comparison.csv"
    )
    args = parser.parse_args()
    row = evaluate_xgb(
        args.xgb_run_dir or _latest_run_dir(),
        refit_train_valid=not args.no_refit_train_valid,
    )
    table = pd.DataFrame([row])
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
