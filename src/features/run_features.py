"""
Feature pipeline runner — loads cleaned data, engineers and selects features,
saves the result to data/features.csv.

Usage:
    python src/features/run_features.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from features.engineering import create_features, select_features

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLEANED_PATH = DATA_DIR / "cleaned.csv"
FEATURES_PATH = DATA_DIR / "features.csv"


def main() -> None:
    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {CLEANED_PATH} ...")
    df_cleaned = pd.read_csv(CLEANED_PATH, parse_dates=["LocalTime"])
    print(f"  Shape: {df_cleaned.shape[0]:,} rows x {df_cleaned.shape[1]} columns")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # Sort required before lag computation (per-site time order).
    df_cleaned = df_cleaned.sort_values(
        ["lat", "lon", "LocalTime"]
    ).reset_index(drop=True)

    # ── Create features ───────────────────────────────────────────────────────
    t1 = time.time()
    print(f"\nEngineering features ...")
    df_features = create_features(df_cleaned)
    new_cols = sorted(set(df_features.columns) - set(df_cleaned.columns))
    dropped_cols = sorted(set(df_cleaned.columns) - set(df_features.columns))
    print(f"  Shape: {df_features.shape[0]:,} rows x {df_features.shape[1]} columns")
    print(f"  Added   ({len(new_cols)}): {new_cols}")
    print(f"  Dropped ({len(dropped_cols)}): {dropped_cols}")
    print(f"  Elapsed: {time.time() - t1:.1f}s")

    # ── Select features ───────────────────────────────────────────────────────
    t2 = time.time()
    print(f"\nSelecting features ...")
    selected_names, df_selected = select_features(df_features)
    print(f"  Shape: {df_selected.shape[0]:,} rows x {df_selected.shape[1]} columns")
    print(f"  Elapsed: {time.time() - t2:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Before (cleaned) : {df_cleaned.shape[1]:>3} columns")
    print(f"  After (engineered): {df_features.shape[1]:>3} columns")
    print(f"  After (selected) : {df_selected.shape[1]:>3} columns")
    print(f"\n  Kept features ({len(selected_names)}):")
    for name in sorted(selected_names):
        print(f"    {name}")

    # ── Save ──────────────────────────────────────────────────────────────────
    t3 = time.time()
    print(f"\nSaving to {FEATURES_PATH} ...")
    df_selected.to_csv(FEATURES_PATH, index=False)
    print(f"  Saved {len(df_selected):,} rows x {len(df_selected.columns)} columns")
    print(f"  Elapsed: {time.time() - t3:.1f}s")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
