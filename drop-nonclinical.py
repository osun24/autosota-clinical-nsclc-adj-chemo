#!/usr/bin/env python3
"""Create clinical-covariate-only train and validation CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SURVIVAL_COLUMNS = ["OS_STATUS", "OS_MONTHS"]
CLINICAL_COLUMNS = [
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
OUTPUT_COLUMNS = SURVIVAL_COLUMNS + CLINICAL_COLUMNS


def clinical_only(source: Path, destination: Path) -> tuple[int, int]:
    """Select the fixed analysis schema and write it without an index."""
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in OUTPUT_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    frame = pd.read_csv(source, usecols=OUTPUT_COLUMNS)[OUTPUT_COLUMNS]
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return len(frame), len(header) - len(OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clinical-only train and validation sets."
    )
    parser.add_argument("--train", type=Path, default=Path("affyfRMATrain.csv"))
    parser.add_argument(
        "--validation", type=Path, default=Path("affyfRMAValidation.csv")
    )
    parser.add_argument(
        "--train-out", type=Path, default=Path("clinicalTrain.csv")
    )
    parser.add_argument(
        "--validation-out", type=Path, default=Path("clinicalValidation.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (
        (args.train, args.train_out),
        (args.validation, args.validation_out),
    )
    for source, destination in outputs:
        rows, removed = clinical_only(source, destination)
        print(
            f"Wrote {destination}: {rows} rows, {len(OUTPUT_COLUMNS)} columns "
            f"({removed} non-analysis columns removed)"
        )


if __name__ == "__main__":
    main()
