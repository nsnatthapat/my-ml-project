"""
MLflow training pipeline for solar PV generation forecasting.

Trains two model configs and logs everything to MLflow:
  1. baseline  — default LightGBM hyperparameters
  2. tuned_best — best hyperparameters from models/best_params.json (Optuna output)

For each run, MLflow records:
  - Parameters : model name, all hyperparameters
  - Metrics    : MAE, RMSE, R² on both train and test sets
  - Artifact   : serialised model (joblib .pkl)

The tuned model is additionally saved as models/production_model.pkl.

Usage:
    # Start the tracking server first (run once in a separate terminal):
    mlflow server --host 127.0.0.1 --port 5000

    # Then run this script:
    python src/models/run_training.py

    # View all runs in the browser:
    open http://localhost:5000
"""

import json
import time
import joblib
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

FEATURES_PATH      = DATA_DIR   / "features.csv"
BEST_PARAMS_PATH   = MODELS_DIR / "best_params.json"
PRODUCTION_MODEL   = MODELS_DIR / "production_model.pkl"

TIME_COL     = "LocalTime"
TARGET_COL   = "Power(MW)"
LAG1_COL     = "power_lag_1"
_DROP_COLS   = {TIME_COL, TARGET_COL, "prefix", "pv_type", "interval"}
EXPERIMENT   = "solar-pv-forecasting"
RANDOM_STATE = 42
N_CV_SPLITS  = 5
CV_SAMPLE    = 500_000   # rows used for CV scoring (time-ordered subsample)

# ── MLflow tracking URI (local file store at project root) ────────────────────
TRACKING_URI = "sqlite:///" + str(Path(__file__).resolve().parents[2] / "mlflow.db")


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
    X: pd.DataFrame, y: pd.Series, n: int
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) > n:
        start = (len(X) - n) // 2
        return X.iloc[start:start + n], y.iloc[start:start + n]
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str
) -> dict:
    return {
        f"{prefix}_mae" : float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_r2"  : float(r2_score(y_true, y_pred)),
    }


def persistence_rmse(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    mask = X_test[LAG1_COL].notna()
    return float(np.sqrt(mean_squared_error(
        y_test[mask].values, X_test.loc[mask, LAG1_COL].values
    )))


# ── Model configs ─────────────────────────────────────────────────────────────

def load_configs() -> list[dict]:
    """
    Return list of {name, params} dicts — one per training run.
    baseline    : sensible defaults matching compare_models.py results.
    tuned_best  : Optuna-optimised params from models/best_params.json.
    """
    baseline_params = {
        "n_estimators"      : 300,
        "learning_rate"     : 0.05,
        "num_leaves"        : 63,
        "subsample"         : 0.8,
        "colsample_bytree"  : 0.8,
        "min_child_samples" : 20,
        "reg_alpha"         : 0.0,
        "reg_lambda"        : 0.0,
        "min_split_gain"    : 0.0,
        "max_depth"         : -1,
    }

    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_PARAMS_PATH} not found. "
            "Run src/models/tuning.py first to generate best_params.json."
        )
    with open(BEST_PARAMS_PATH) as f:
        tuned_params = json.load(f)["hyperparameters"]

    return [
        {"name": "baseline",   "params": baseline_params},
        {"name": "tuned_best", "params": tuned_params},
    ]


# ── Single training run ───────────────────────────────────────────────────────

def train_and_log(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    persist_rmse: float,
) -> tuple[lgb.LGBMRegressor, dict]:
    """
    Train one model config, log everything to MLflow, return (model, test_metrics).
    """
    name   = config["name"]
    params = config["params"]

    with mlflow.start_run(run_name=name):
        # ── Log parameters ────────────────────────────────────────────────────
        mlflow.log_param("model_name",   "LightGBM")
        mlflow.log_param("config",       name)
        mlflow.log_param("train_rows",   len(X_train))
        mlflow.log_param("test_rows",    len(X_test))
        mlflow.log_param("n_features",   len(X_train.columns))
        mlflow.log_param("features",     ",".join(sorted(X_train.columns)))
        for k, v in params.items():
            mlflow.log_param(k, v)

        # ── 5-fold CV score ───────────────────────────────────────────────────
        model_cv = lgb.LGBMRegressor(
            **params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        X_cv, y_cv = cv_subsample(X_train, y_train, CV_SAMPLE)
        tscv       = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        scores     = cross_val_score(
            model_cv, X_cv, y_cv,
            cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1
        )
        cv_rmse = float(-scores.mean())
        cv_std  = float(scores.std())
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_std",  cv_std)

        # ── Final fit on full training set ────────────────────────────────────
        t0    = time.time()
        model = lgb.LGBMRegressor(
            **params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time_s", train_time)

        # ── Train metrics ─────────────────────────────────────────────────────
        train_metrics = compute_metrics(
            y_train.values, model.predict(X_train), prefix="train"
        )
        mlflow.log_metrics(train_metrics)

        # ── Test metrics ──────────────────────────────────────────────────────
        test_metrics = compute_metrics(
            y_test.values, model.predict(X_test), prefix="test"
        )
        mlflow.log_metrics(test_metrics)

        # ── Improvement over persistence ──────────────────────────────────────
        rmse_gain_pct = (persist_rmse - test_metrics["test_rmse"]) / persist_rmse * 100
        mlflow.log_metric("rmse_gain_vs_persistence_pct", rmse_gain_pct)

        # ── Log model artifact via joblib ─────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / f"{name}_model.pkl"
            joblib.dump(
                {
                    "model"       : model,
                    "features"    : X_train.columns.tolist(),
                    "params"      : params,
                    "cv_rmse"     : cv_rmse,
                    "test_metrics": test_metrics,
                },
                artifact_path,
            )
            mlflow.log_artifact(str(artifact_path), artifact_path="model")

        # ── Log feature importances ───────────────────────────────────────────
        importances = dict(zip(
            X_train.columns,
            model.feature_importances_ / model.feature_importances_.sum()
        ))
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            mlflow.log_metric(f"feat_imp_{feat}", round(imp, 6))

        run_id = mlflow.active_run().info.run_id

    return model, test_metrics, run_id


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    t_total = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    print(f"MLflow tracking URI : {TRACKING_URI}")
    print(f"Experiment          : {EXPERIMENT}")
    print(f"View runs at        : http://localhost:5000  (after starting server)")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading {FEATURES_PATH} ...")
    train_df, test_df = load_and_split(test_size=0.2)
    X_train, y_train  = prepare_xy(train_df)
    X_test,  y_test   = prepare_xy(test_df)

    print(f"  Train : {train_df[TIME_COL].min()} → {train_df[TIME_COL].max()}  "
          f"({len(X_train):,} rows)")
    print(f"  Test  : {test_df[TIME_COL].min()} → {test_df[TIME_COL].max()}  "
          f"({len(X_test):,} rows)")

    persist_rmse = persistence_rmse(X_test, y_test)
    print(f"  Persistence RMSE  : {persist_rmse:.4f}")

    # ── Train all configs ──────────────────────────────────────────────────────
    configs = load_configs()
    results = []
    best_model  = None
    best_run_id = None

    for cfg in configs:
        print(f"\n{'─' * 60}")
        print(f"Config: {cfg['name']}")
        print(f"  Params: {cfg['params']}")
        print(f"  Running 5-fold CV + final fit ...")

        model, test_m, run_id = train_and_log(
            cfg, X_train, y_train, X_test, y_test, persist_rmse
        )

        results.append({
            "config"    : cfg["name"],
            "run_id"    : run_id,
            "test_rmse" : test_m["test_rmse"],
            "test_mae"  : test_m["test_mae"],
            "test_r2"   : test_m["test_r2"],
        })
        print(f"  MLflow run_id: {run_id}")
        print(f"  Test  MAE={test_m['test_mae']:.4f}  "
              f"RMSE={test_m['test_rmse']:.4f}  "
              f"R2={test_m['test_r2']:.4f}")

        if cfg["name"] == "tuned_best":
            best_model  = model
            best_run_id = run_id

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("TRAINING RUN SUMMARY\n")
    print(f"  {'Config':<14}  {'Test MAE':>9}  {'Test RMSE':>10}  {'Test R2':>8}  Run ID")
    print(f"  {'─'*14}  {'─'*9}  {'─'*10}  {'─'*8}  {'─'*32}")
    for r in results:
        print(f"  {r['config']:<14}  {r['test_mae']:>9.4f}  "
              f"{r['test_rmse']:>10.4f}  {r['test_r2']:>8.4f}  {r['run_id']}")

    # ── Save production model ─────────────────────────────────────────────────
    if best_model is not None:
        joblib.dump(
            {
                "model"   : best_model,
                "features": X_train.columns.tolist(),
                "config"  : "tuned_best",
                "run_id"  : best_run_id,
            },
            PRODUCTION_MODEL,
        )
        print(f"\nProduction model saved → {PRODUCTION_MODEL}")
        print(f"  Source run : {best_run_id}")

    print(f"\nTotal elapsed : {time.time()-t_total:.1f}s")
    print(f"\nStart the MLflow UI to explore runs:")
    print(f"  mlflow server --host 127.0.0.1 --port 5000")
    print(f"  open http://localhost:5000")


if __name__ == "__main__":
    run()
