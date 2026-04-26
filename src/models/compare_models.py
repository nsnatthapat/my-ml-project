"""
Model comparison for multi-step ahead solar PV generation forecasting.

Three models are trained and compared against the persistence baseline:

  1. Ridge Regression   — regularised linear model; handles correlated lag features
  2. LightGBM           — fast gradient boosting; state-of-the-art for tabular TS
  3. XGBoost            — gradient boosting with different regularisation strategy

Evaluation protocol (time series correct):
  - 80/20 chronological split (no shuffling)
  - 5-fold TimeSeriesSplit cross-validation within the training set
  - Final evaluation on the held-out 20% test set

Usage:
    python src/models/compare_models.py
    python src/models/compare_models.py --cv-sample 500000
"""

import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

FEATURES_PATH = DATA_DIR / "features.csv"
TIME_COL      = "LocalTime"
TARGET_COL    = "Power(MW)"
LAG1_COL      = "power_lag_1"
_DROP_COLS    = {TIME_COL, TARGET_COL, "prefix", "pv_type", "interval"}

N_CV_SPLITS   = 5
RANDOM_STATE  = 42


# ── Model definitions ─────────────────────────────────────────────────────────
#
# WHY these three models for solar PV time series forecasting:
#
# 1. Ridge Regression
#    Lag features are highly correlated (lag_1 / lag_12 / lag_24 share r ≈ 0.97).
#    OLS is numerically unstable with near-multicollinear features; the L2 penalty
#    shrinks all coefficients toward zero, giving a stable and well-calibrated
#    linear estimate. Acts as a stronger linear baseline than plain LinearRegression.
#
# 2. LightGBM (Gradient Boosted Trees — leaf-wise)
#    Tree models capture non-linear interactions that linear models miss: e.g. the
#    relationship between capacity_mw and daytime output is not additive. LightGBM's
#    leaf-wise growth and histogram binning are specifically optimised for large
#    datasets (8M rows) and mixed-magnitude features. It implicitly handles the
#    zero-inflated target (70% zeros at night) via asymmetric leaf assignments.
#    No feature scaling required; native handling of different feature scales.
#
# 3. XGBoost (Gradient Boosted Trees — depth-wise)
#    Provides a second gradient boosting perspective with different regularisation:
#    depth-wise growth (more conservative trees) + both L1 and L2 penalties.
#    tree_method='hist' makes it practical at 8M rows. Comparing against LightGBM
#    tests whether the more aggressive leaf-wise growth or the conservative depth-
#    wise approach generalises better on unseen future data.

def build_models() -> dict[str, Pipeline | object]:
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ── Data helpers ──────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    cutoff_time = df[TIME_COL].iloc[int(len(df) * (1 - test_size))]
    return (
        df[df[TIME_COL] <  cutoff_time].copy(),
        df[df[TIME_COL] >= cutoff_time].copy(),
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    meta_drop = list((_DROP_COLS - {TARGET_COL}) & set(df.columns))
    df = df.drop(columns=meta_drop).dropna()
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE" : mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2"  : r2_score(y_true, y_pred),
    }


def persistence_metrics(X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    mask = X_test[LAG1_COL].notna() & y_test.notna()
    return test_metrics(y_test[mask].values, X_test.loc[mask, LAG1_COL].values)


# ── CV helper ─────────────────────────────────────────────────────────────────

def run_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_sample: int | None,
) -> tuple[float, float]:
    """
    5-fold TimeSeriesSplit CV on a (optionally time-stratified) subsample.
    Returns (mean_rmse, std_rmse).
    """
    if cv_sample and len(X) > cv_sample:
        # Take a contiguous time-ordered slice from the middle of training
        # so the CV folds still see a realistic temporal structure.
        start = (len(X) - cv_sample) // 2
        X_cv = X.iloc[start : start + cv_sample]
        y_cv = y.iloc[start : start + cv_sample]
    else:
        X_cv, y_cv = X, y

    tscv   = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    scores = cross_val_score(
        model, X_cv, y_cv,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,       # avoid forking issues with LightGBM/XGBoost inner threads
    )
    return float(-scores.mean()), float(scores.std())


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(rows: list[dict]) -> None:
    print(f"\n{'─' * 82}")
    print(
        f"  {'Model':<22}  {'CV RMSE':>10}  {'CV std':>8}  "
        f"{'Test MAE':>9}  {'Test RMSE':>10}  {'Test R2':>8}  {'Train (s)':>9}"
    )
    print(f"  {'─' * 22}  {'─' * 10}  {'─' * 8}  {'─' * 9}  {'─' * 10}  {'─' * 8}  {'─' * 9}")
    for r in rows:
        print(
            f"  {r['Model']:<22}  {r['CV RMSE']:>10.4f}  {r['CV std']:>8.4f}  "
            f"{r['Test MAE']:>9.4f}  {r['Test RMSE']:>10.4f}  {r['Test R2']:>8.4f}  "
            f"{r['Train (s)']:>9.1f}"
        )
    print(f"{'─' * 82}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cv_sample: int | None = 500_000) -> None:
    t_total = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    # ── Load & split ──────────────────────────────────────────────────────────
    print(f"Loading {FEATURES_PATH} ...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=[TIME_COL])
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    train_df, test_df = chronological_split(df, test_size=0.2)
    print(f"\nChronological split:")
    print(f"  Train: {train_df[TIME_COL].min()} → {train_df[TIME_COL].max()}  ({len(train_df):,} rows)")
    print(f"  Test : {test_df[TIME_COL].min()} → {test_df[TIME_COL].max()}  ({len(test_df):,} rows)")

    X_train, y_train = prepare_xy(train_df)
    X_test,  y_test  = prepare_xy(test_df)
    print(f"  Features ({len(X_train.columns)}): {sorted(X_train.columns.tolist())}")

    # ── Persistence baseline (no training needed) ─────────────────────────────
    print(f"\n{'─' * 56}")
    print("Persistence baseline (ŷ = lag-1, no training):")
    p_metrics = persistence_metrics(X_test, y_test)
    rows = [{
        "Model"    : "Persistence (lag-1)",
        "CV RMSE"  : float("nan"),
        "CV std"   : float("nan"),
        "Test MAE" : p_metrics["MAE"],
        "Test RMSE": p_metrics["RMSE"],
        "Test R2"  : p_metrics["R2"],
        "Train (s)": 0.0,
    }]
    print(f"  MAE={p_metrics['MAE']:.4f}  RMSE={p_metrics['RMSE']:.4f}  R2={p_metrics['R2']:.4f}")

    # ── Train & evaluate each model ───────────────────────────────────────────
    models = build_models()
    cv_note = f" (subsampled to {cv_sample:,} rows)" if cv_sample else ""

    for name, model in models.items():
        print(f"\n{'─' * 56}")
        print(f"Model: {name}")

        # 5-fold TimeSeriesSplit CV
        print(f"  Running {N_CV_SPLITS}-fold TimeSeriesSplit CV{cv_note} ...")
        t_cv = time.time()
        cv_mean, cv_std = run_cv(model, X_train, y_train, cv_sample)
        print(f"  CV RMSE: {cv_mean:.4f} ± {cv_std:.4f}  ({time.time()-t_cv:.1f}s)")

        # Final fit on full training set
        print(f"  Training on full train set ({len(X_train):,} rows) ...")
        t_fit = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t_fit
        print(f"  Training time: {train_time:.1f}s")

        # Test set evaluation
        y_pred = model.predict(X_test)
        m = test_metrics(y_test.values, y_pred)
        print(f"  Test  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

        # Save model
        save_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(
            {"model": model, "features": X_train.columns.tolist(),
             "task": "time_series_regression", "metrics": m},
            save_path,
        )
        print(f"  Saved → {save_path.name}")

        rows.append({
            "Model"    : name,
            "CV RMSE"  : cv_mean,
            "CV std"   : cv_std,
            "Test MAE" : m["MAE"],
            "Test RMSE": m["RMSE"],
            "Test R2"  : m["R2"],
            "Train (s)": train_time,
        })

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'═' * 82}")
    print("COMPARISON TABLE  (test window: Oct 20 – Dec 31 2006)\n")
    print_comparison(rows)

    # ── Winner analysis ───────────────────────────────────────────────────────
    scored = [r for r in rows if not np.isnan(r["CV RMSE"])]
    best   = min(scored, key=lambda r: r["Test RMSE"])
    runner = sorted(scored, key=lambda r: r["Test RMSE"])[1]

    print(f"\nBest model: {best['Model']}")
    print(f"  Test RMSE {best['Test RMSE']:.4f}  vs persistence {p_metrics['RMSE']:.4f}")
    rmse_gain = (p_metrics["RMSE"] - best["Test RMSE"]) / p_metrics["RMSE"] * 100
    mae_gain  = (p_metrics["MAE"]  - best["Test MAE"])  / p_metrics["MAE"]  * 100
    print(f"  Beats persistence by  RMSE {rmse_gain:+.1f}%  |  MAE {mae_gain:+.1f}%")
    print(f"  Runner-up: {runner['Model']}  (RMSE {runner['Test RMSE']:.4f})")

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and compare models for solar PV time series forecasting"
    )
    parser.add_argument(
        "--cv-sample",
        type=int,
        default=500_000,
        metavar="N",
        help="Rows used for CV scoring (default 500k — use 0 for full training set)",
    )
    args = parser.parse_args()
    run(cv_sample=args.cv_sample or None)
