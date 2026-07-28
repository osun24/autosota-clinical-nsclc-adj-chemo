"""Human-only sealed test comparison across arena artifacts.

This script may load `affyfRMATest.csv`. Do not run it from any autonomous loop.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

import finalize as finalize_xgb

REPO_ROOT = Path(__file__).resolve().parent


def _has_completed_run(runs_dir: Path) -> bool:
    return runs_dir.exists() and any(runs_dir.glob("*/metadata.json"))


def _load_hyphen_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-only sealed test comparison across arenas.")
    parser.add_argument("--xgb-run-dir", type=Path, default=None)
    parser.add_argument("--rsf-run-dir", type=Path, default=None)
    parser.add_argument("--deepsurv-run-dir", type=Path, default=None)
    parser.add_argument("--no-refit-train-valid", action="store_true")
    parser.add_argument("--hand-tuned-row", type=Path, default=None, help="Optional CSV with user's hand-tuned result row.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "final_all_comparison.csv")
    args = parser.parse_args()

    refit = not args.no_refit_train_valid

    rows = []
    if args.xgb_run_dir is not None or _has_completed_run(finalize_xgb.XGB_RUNS_DIR):
        xgb_dir = args.xgb_run_dir or finalize_xgb._latest_run_dir(finalize_xgb.XGB_RUNS_DIR)
        rows.append(finalize_xgb.evaluate_xgb(xgb_dir, refit_train_valid=refit))
    rsf_runs_dir = REPO_ROOT / "rsf_arena" / "runs"
    if args.rsf_run_dir is not None or _has_completed_run(rsf_runs_dir):
        finalize_rsf = _load_hyphen_module("finalize-rsf.py", "finalize_rsf")
        rsf_dir = args.rsf_run_dir or finalize_rsf.latest_run_dir()
        rows.append(finalize_rsf.evaluate_rsf(rsf_dir, refit_train_valid=refit))
    deepsurv_runs_dir = REPO_ROOT / "deepsurv_arena" / "runs"
    if args.deepsurv_run_dir is not None or _has_completed_run(deepsurv_runs_dir):
        finalize_deepsurv = _load_hyphen_module(
            "finalize-deepsurv.py", "finalize_deepsurv"
        )
        deepsurv_dir = args.deepsurv_run_dir or finalize_deepsurv.latest_run_dir()
        rows.append(finalize_deepsurv.evaluate_deepsurv(deepsurv_dir, refit_train_valid=refit))

    table = pd.DataFrame(rows)
    if args.hand_tuned_row is not None:
        hand = pd.read_csv(args.hand_tuned_row)
        table = pd.concat([table, hand], ignore_index=True, sort=False)
    print(table.to_string(index=False))
    table.to_csv(args.out, index=False)
    print(f"\nWrote combined comparison table to {args.out}")


if __name__ == "__main__":
    main()
