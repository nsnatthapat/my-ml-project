"""
Baseline models for multi-step ahead solar PV generation forecasting.

Two baselines are trained and compared:
  1. Persistence model  — naive: ŷ(t) = y(t-1)  [power_lag_1]
                          The minimum bar any learned model must beat.
  2. Linear model       — LinearRegression on all engineered features.

Both use a CHRONOLOGICAL train/test split (no shuffling):
  - Train: first 80% of timesteps per site
  - Test : last  20% of timesteps per site
This prevents future data from leaking into the training window.

Usage:
    python src/models/baseline.py
    python src/models/baseline.py --test-size 0.2
"""

import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

FEATURES_PATH = DATA_DIR   / "features.csv"
MODEL_PATH    = MODELS_DIR / "baseline.pkl"

TARGET_COL = "Power(MW)"
TIME_COL   = "LocalTime"
LAG1_COL   = "power_lag_1"          # persistence forecast source

# Columns to exclude from the feature matrix X
_DROP_COLS = {TIME_COL, TARGET_COL, "prefix", "pv_type", "interval"}


# ── Metrics ───────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"model": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def print_metrics_table(results: list[dict]) -> None:
    header = f"  {'Model':<22}  {'MAE':>10}  {'RMSE':>10}  {'R2':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['model']:<22}  "
            f"{r['MAE']:>10.4f}  "
            f"{r['RMSE']:>10.4f}  "
            f"{r['R2']:>10.4f}"
        )


# ── Chronological split ───────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Global chronological split on the TIME_COL timestamp.

    All rows before the cutoff timestamp → train.
    All rows from the cutoff timestamp onward → test.

    This mirrors real deployment: the model is trained on historical data
    and evaluated on genuinely unseen future data across all sites simultaneously.
    No shuffling — future rows never appear in the training window.
    """
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    cutoff_idx  = int(len(df) * (1 - test_size))
    cutoff_time = df[TIME_COL].iloc[cutoff_idx]

    train = df[df[TIME_COL] <  cutoff_time].copy()
    test  = df[df[TIME_COL] >= cutoff_time].copy()
    return train, test


# ── Prepare feature matrix ────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop non-feature columns and NaN rows, return (X, y)."""
    # Drop metadata columns (everything in _DROP_COLS except the target itself)
    meta_drop = list((_DROP_COLS - {TARGET_COL}) & set(df.columns))
    df = df.drop(columns=meta_drop).dropna()
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(test_size: float = 0.2) -> None:
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {FEATURES_PATH} ...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=[TIME_COL])
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    # ── Chronological split ───────────────────────────────────────────────────
    print(f"\nSplitting chronologically (train={1-test_size:.0%} / test={test_size:.0%}) per site ...")
    train_df, test_df = chronological_split(df, test_size=test_size)

    # Time boundaries for transparency
    train_end = train_df[TIME_COL].max()
    test_start = test_df[TIME_COL].min()
    print(f"  Train: {train_df[TIME_COL].min()}  →  {train_end}   ({len(train_df):,} rows)")
    print(f"  Test : {test_start}  →  {test_df[TIME_COL].max()}   ({len(test_df):,} rows)")

    # ── Prepare X / y ─────────────────────────────────────────────────────────
    X_train, y_train = prepare_xy(train_df)
    X_test,  y_test  = prepare_xy(test_df)
    print(f"\n  Features ({len(X_train.columns)}): {sorted(X_train.columns.tolist())}")
    print(f"  Train rows (after NaN drop): {len(X_train):,}")
    print(f"  Test  rows (after NaN drop): {len(X_test):,}")

    results = []

    # ── Baseline 1: Persistence model ─────────────────────────────────────────
    # Predict ŷ(t) = y(t-1)  — the last observed value.
    # This is the canonical naive benchmark for time series. Any model that
    # cannot beat persistence is not useful for operational forecasting.
    print(f"\n{'─' * 56}")
    print("Baseline 1: Persistence  (ŷ(t) = y(t-1))")
    y_persist = X_test[LAG1_COL]                     # power_lag_1 is already in X
    valid_mask = y_persist.notna() & y_test.notna()  # exclude warm-up NaNs
    persist_metrics = regression_metrics(
        y_test[valid_mask].values,
        y_persist[valid_mask].values,
        "Persistence (lag-1)",
    )
    results.append(persist_metrics)

    # ── Baseline 2: Linear regression ─────────────────────────────────────────
    # StandardScaler + LinearRegression. The scaler is important because
    # features span very different magnitudes (capacity_mw in 0.1–109,
    # hour in 0–23, lon in -123 to -117).
    print(f"\nBaseline 2: LinearRegression")
    t1 = time.time()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    print(f"  Training elapsed: {time.time() - t1:.1f}s")

    y_pred = pipeline.predict(X_test)
    lr_metrics = regression_metrics(y_test.values, y_pred, "LinearRegression")
    results.append(lr_metrics)

    # ── Evaluation table ──────────────────────────────────────────────────────
    print(f"\n{'─' * 56}")
    print("Test set evaluation (chronological 20% holdout):\n")
    print_metrics_table(results)

    # Improvement over persistence
    delta_mae  = persist_metrics["MAE"]  - lr_metrics["MAE"]
    delta_rmse = persist_metrics["RMSE"] - lr_metrics["RMSE"]
    print(f"\n  LinearRegression vs Persistence:")
    print(f"    MAE  improvement : {delta_mae:+.4f} MW  "
          f"({'better' if delta_mae > 0 else 'worse'})")
    print(f"    RMSE improvement : {delta_rmse:+.4f} MW  "
          f"({'better' if delta_rmse > 0 else 'worse'})")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving model to {MODEL_PATH} ...")
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": X_train.columns.tolist(),
            "task": "time_series_regression",
            "test_size": test_size,
            "train_end": str(train_end),
            "test_start": str(test_start),
            "metrics": {r["model"]: r for r in results},
        },
        MODEL_PATH,
    )
    print(f"  Saved.")
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate time-series baseline models"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of each site's series to hold out as test (default: 0.2)",
    )
    args = parser.parse_args()
    run(test_size=args.test_size)
