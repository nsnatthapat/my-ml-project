"""
Hyperparameter tuning for LightGBM on solar PV generation forecasting.

Uses Optuna (TPE sampler) with 30 trials, 5-fold TimeSeriesSplit CV.
Each trial is logged with its hyperparameters and CV RMSE score.
The best parameters are saved to models/best_params.json and the final
model (re-trained on the full training set) to models/tuned_model.pkl.

Usage:
    python src/models/tuning.py
    python src/models/tuning.py --trials 50 --cv-sample 300000
"""

import json
import sys
import time
import argparse
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

FEATURES_PATH   = DATA_DIR   / "features.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
TUNED_MODEL_PATH = MODELS_DIR / "tuned_model.pkl"

TIME_COL     = "LocalTime"
TARGET_COL   = "Power(MW)"
_DROP_COLS   = {TIME_COL, TARGET_COL, "prefix", "pv_type", "interval"}
N_CV_SPLITS  = 5
RANDOM_STATE = 42

# Silence LightGBM iteration logs during CV
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_and_split(
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(FEATURES_PATH, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    cutoff = df[TIME_COL].iloc[int(len(df) * (1 - test_size))]
    return df[df[TIME_COL] < cutoff].copy(), df[df[TIME_COL] >= cutoff].copy()


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    meta = list((_DROP_COLS - {TARGET_COL}) & set(df.columns))
    df   = df.drop(columns=meta).dropna()
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def cv_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    n: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a contiguous time-ordered slice of size n (or full data if n is None)."""
    if n and len(X) > n:
        start = (len(X) - n) // 2
        return X.iloc[start:start + n], y.iloc[start:start + n]
    return X, y


# ── Search space ──────────────────────────────────────────────────────────────

def suggest_params(trial: optuna.Trial) -> dict:
    """
    LightGBM hyperparameter search space.

    Grouped by what each parameter controls:

    Tree structure
      num_leaves      — model capacity; higher = more complex, risk of overfit
      max_depth       — hard cap on tree depth (-1 = no limit)
      min_child_samples — minimum rows per leaf; controls regularisation

    Learning rate & boosting rounds
      learning_rate   — step size; lower = better generalisation, needs more trees
      n_estimators    — number of boosting rounds; paired with learning_rate

    Stochastic regularisation
      subsample       — fraction of rows sampled per tree (like RF bagging)
      colsample_bytree — fraction of features sampled per tree

    L1 / L2 regularisation
      reg_alpha       — L1 penalty (sparsity); helps with irrelevant features
      reg_lambda      — L2 penalty (magnitude shrinkage); default LightGBM behaviour

    Minimum gain
      min_split_gain  — minimum loss reduction to split a node; prunes weak splits
    """
    return {
        "num_leaves"        : trial.suggest_int("num_leaves", 20, 300),
        "max_depth"         : trial.suggest_int("max_depth", 3, 12),
        "min_child_samples" : trial.suggest_int("min_child_samples", 10, 200),
        "learning_rate"     : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators"      : trial.suggest_int("n_estimators", 100, 600),
        "subsample"         : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain"    : trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


# ── Objective ─────────────────────────────────────────────────────────────────

def make_objective(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    trial_log: list[dict],
) -> callable:
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        model  = lgb.LGBMRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

        t0     = time.time()
        scores = cross_val_score(
            model, X_cv, y_cv,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        elapsed = time.time() - t0
        cv_rmse = float(-scores.mean())
        cv_std  = float(scores.std())

        # Log every trial
        entry = {
            "trial"  : trial.number,
            "cv_rmse": round(cv_rmse, 6),
            "cv_std" : round(cv_std,  6),
            "elapsed": round(elapsed, 1),
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in params.items()},
        }
        trial_log.append(entry)

        status = "★ best" if cv_rmse == min(e["cv_rmse"] for e in trial_log) else ""
        print(
            f"  Trial {trial.number:>3}  CV RMSE={cv_rmse:.4f} ±{cv_std:.4f}"
            f"  lr={params['learning_rate']:.4f}  leaves={params['num_leaves']}"
            f"  est={params['n_estimators']}  ({elapsed:.1f}s)  {status}"
        )
        return cv_rmse

    return objective


# ── Final evaluation ──────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE" : float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2"  : float(r2_score(y_true, y_pred)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n_trials: int = 30, cv_sample: int | None = 500_000) -> None:
    t_total = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {FEATURES_PATH} ...")
    train_df, test_df = load_and_split(test_size=0.2)
    X_train, y_train  = prepare_xy(train_df)
    X_test,  y_test   = prepare_xy(test_df)

    print(f"  Train: {len(train_df[TIME_COL].unique())} timestamps  "
          f"({len(X_train):,} rows after NaN drop)")
    print(f"  Test : {len(X_test):,} rows")
    print(f"  Features ({len(X_train.columns)}): {sorted(X_train.columns.tolist())}")

    X_cv, y_cv = cv_subsample(X_train, y_train, cv_sample)
    cv_note = f"(subsampled to {len(X_cv):,} rows)" if len(X_cv) < len(X_train) else "(full train set)"
    print(f"\n  CV data: {len(X_cv):,} rows {cv_note}")

    # ── Optuna study ───────────────────────────────────────────────────────────
    print(f"\nStarting Optuna study: {n_trials} trials, "
          f"{N_CV_SPLITS}-fold TimeSeriesSplit CV")
    print(f"{'─' * 72}")

    trial_log: list[dict] = []
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name="lgbm_solar_pv",
    )
    study.optimize(
        make_objective(X_cv, y_cv, trial_log),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # ── Best trial ─────────────────────────────────────────────────────────────
    best = study.best_trial
    best_params = best.params
    best_cv_rmse = best.value

    print(f"\n{'─' * 72}")
    print(f"Best trial: #{best.number}  CV RMSE={best_cv_rmse:.4f}")
    print(f"\nBest hyperparameters:")
    for k, v in sorted(best_params.items()):
        print(f"  {k:<22}: {v}")

    # ── Save best params ───────────────────────────────────────────────────────
    payload = {
        "model"         : "LightGBM",
        "best_trial"    : best.number,
        "cv_rmse"       : round(best_cv_rmse, 6),
        "n_trials"      : n_trials,
        "cv_splits"     : N_CV_SPLITS,
        "hyperparameters": best_params,
        "all_trials"    : trial_log,
    }
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved best params → {BEST_PARAMS_PATH}")

    # ── Train final model on full training set ─────────────────────────────────
    print(f"\nTraining final model on full training set ({len(X_train):,} rows) ...")
    t0 = time.time()
    final_model = lgb.LGBMRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # ── Test set evaluation ────────────────────────────────────────────────────
    y_pred = final_model.predict(X_test)
    m = evaluate(y_test.values, y_pred)

    print(f"\nTest set evaluation (chronological 20% holdout):")
    print(f"{'─' * 40}")
    print(f"  {'Metric':<8}  {'Tuned LightGBM':>16}  {'Baseline LightGBM':>18}")
    print(f"  {'─'*8}  {'─'*16}  {'─'*18}")
    # Baseline numbers from compare_models.py for reference
    baseline = {"MAE": 0.1858, "RMSE": 1.0969, "R2": 0.9656}
    for metric in ("MAE", "RMSE", "R2"):
        delta = m[metric] - baseline[metric]
        direction = "better" if (metric == "R2" and delta > 0) or (metric != "R2" and delta < 0) else "worse"
        print(f"  {metric:<8}  {m[metric]:>16.4f}  {baseline[metric]:>14.4f}  "
              f"({delta:+.4f} {direction})")

    # ── Save tuned model ───────────────────────────────────────────────────────
    joblib.dump(
        {
            "model"     : final_model,
            "features"  : X_train.columns.tolist(),
            "params"    : best_params,
            "cv_rmse"   : best_cv_rmse,
            "test_metrics": m,
        },
        TUNED_MODEL_PATH,
    )
    print(f"\nSaved tuned model → {TUNED_MODEL_PATH}")
    print(f"Total elapsed: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune LightGBM with Optuna for solar PV time series forecasting"
    )
    parser.add_argument("--trials",    type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--cv-sample", type=int, default=500_000, metavar="N",
                        help="Rows used for CV (default: 500k; 0 = full train set)")
    args = parser.parse_args()
    run(n_trials=args.trials, cv_sample=args.cv_sample or None)
