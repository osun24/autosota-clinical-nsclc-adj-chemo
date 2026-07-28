"""Human-only sealed-test evaluation for clinical RSF artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_data import alignment_rmst_difference
from rsf_arena import prepare
from rsf_arena import train

REPO_ROOT = Path(__file__).resolve().parent
TEST_CSV = REPO_ROOT / "affyfRMATest.csv"
RSF_RUNS_DIR = REPO_ROOT / "rsf_arena" / "runs"


def latest_run_dir(runs_dir: Path = RSF_RUNS_DIR) -> Path:
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


def evaluate_rsf(
    run_dir: Path,
    refit_train_valid: bool = True,
    km_figure_path: Path | None = None,
) -> dict:
    del km_figure_path
    run_dir = run_dir.resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    feature_names = list(metadata["feature_names"])
    if refit_train_valid:
        train_df, valid_df = prepare.load_train_valid()
        model, feature_names, _ = train.fit_model_from_metadata(
            metadata, train_df, valid_df, refit_train_valid=True
        )
        source = "refit_train_valid"
    else:
        with (run_dir / "rsf_model.pkl").open("rb") as stream:
            model = pickle.load(stream)
        source = "saved_validation_model"
    test_df = _load_test()
    observed = prepare.build_matrix_from_feature_names(
        test_df, feature_names
    )
    risk = prepare.predict_rsf_risk(model, observed)
    treated = test_df.copy()
    treated["Adjuvant Chemo"] = 1
    untreated = test_df.copy()
    untreated["Adjuvant Chemo"] = 0
    recommendation = (
        prepare.predict_rsf_risk(
            model,
            prepare.build_matrix_from_feature_names(treated, feature_names),
        )
        < prepare.predict_rsf_risk(
            model,
            prepare.build_matrix_from_feature_names(untreated, feature_names),
        )
    ).astype(int)
    return {
        "model": "rsf_clinical",
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "model_source": source,
        "test_ci": prepare.cindex(
            risk,
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
        description="Human-only sealed-test evaluation for RSF."
    )
    parser.add_argument("--rsf-run-dir", type=Path)
    parser.add_argument("--no-refit-train-valid", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "final_rsf_comparison.csv"
    )
    args = parser.parse_args()
    row = evaluate_rsf(
        args.rsf_run_dir or latest_run_dir(),
        refit_train_valid=not args.no_refit_train_valid,
    )
    table = pd.DataFrame([row])
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
