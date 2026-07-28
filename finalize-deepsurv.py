"""Human-only sealed-test evaluation for clinical DeepSurv artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deepsurv_arena import prepare
from deepsurv_arena import train

REPO_ROOT = Path(__file__).resolve().parent
TEST_CSV = REPO_ROOT / "affyfRMATest.csv"
DEEPSURV_RUNS_DIR = REPO_ROOT / "deepsurv_arena" / "runs"


def latest_run_dir(runs_dir: Path = DEEPSURV_RUNS_DIR) -> Path:
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


def evaluate_deepsurv(
    run_dir: Path, refit_train_valid: bool = True
) -> dict:
    run_dir = run_dir.resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    feature_names = list(metadata["feature_names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if refit_train_valid:
        train_df, valid_df = prepare.load_train_valid()
        model, transform, feature_names, _ = train.fit_model_from_metadata(
            metadata,
            train_df,
            valid_df,
            refit_train_valid=True,
            device=device,
        )
        source = "refit_train_valid"
    else:
        saved = np.load(run_dir / "tabular_transform.npz")
        transform = {key: saved[key] for key in saved.files}
        params = metadata["deepsurv_params"]
        model = train.DeepSurvMLP(
            len(feature_names), params["hidden_layers"], params["dropout"]
        ).to(device)
        model.load_state_dict(
            torch.load(run_dir / "deepsurv_model.pt", map_location=device)
        )
        model.eval()
        source = "saved_validation_model"
    test_df = _load_test()
    matrix = prepare.apply_tabular_transform(
        prepare.build_matrix_from_feature_names(test_df, feature_names),
        transform,
    )
    risk = prepare.predict_deepsurv_risk(model, matrix, device)
    return {
        "model": "deepsurv_clinical",
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "model_source": source,
        "test_ci": prepare.cindex(
            risk,
            test_df["OS_MONTHS"].to_numpy(float),
            test_df["OS_STATUS"].to_numpy(int),
        ),
        "test_rmst_diff": prepare.compute_alignment_rmst_diff_deepsurv(
            model, test_df, feature_names, transform, device
        ),
        "n_features": len(feature_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-only sealed-test evaluation for DeepSurv."
    )
    parser.add_argument("--deepsurv-run-dir", type=Path)
    parser.add_argument("--no-refit-train-valid", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "final_deepsurv_comparison.csv",
    )
    args = parser.parse_args()
    row = evaluate_deepsurv(
        args.deepsurv_run_dir or latest_run_dir(),
        refit_train_valid=not args.no_refit_train_valid,
    )
    table = pd.DataFrame([row])
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
