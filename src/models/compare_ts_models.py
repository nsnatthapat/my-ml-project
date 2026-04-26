"""
Classical and foundation time-series model comparison.

Target: total regional solar generation (MW), hourly resolution.
All Actual-prefix sites are summed per timestamp then resampled to 1-hour
intervals, creating one univariate time series of 8,760 hourly observations.

Models evaluated:
  ARIMA(2,1,2)              — classical autoregressive baseline
  SARIMA(1,1,1)(1,1,1,24)  — adds daily seasonality (S=24 hours)
  TimesGPT                  — Nixtla foundation model (needs NIXTLA_API_KEY)
  TimesFM                   — Google foundation model (SKIPPED: JAX/lingvo
                               wheel unavailable on macOS ARM; see note below)

Evaluation protocol:
  - Global 80/20 chronological split (same cutoff as compare_models.py)
  - 5-fold walk-forward CV within the training set
  - Final evaluation on held-out test window (Oct 20 – Dec 31 2006)

Usage:
    python src/models/compare_ts_models.py
    NIXTLA_API_KEY=<key> python src/models/compare_ts_models.py

TimesFM note:
    timesfm requires the JAX `lingvo` package which has no macOS ARM wheel.
    To run on Linux / Colab:
        pip install timesfm
        # then uncomment the TimesFM section below
"""

import os
import time
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES_PATH = DATA_DIR / "features.csv"
TIME_COL      = "LocalTime"
TARGET_COL    = "Power(MW)"


# ── Build regional hourly time series ─────────────────────────────────────────

def build_regional_series(features_path: Path) -> pd.Series:
    """
    Sum Power(MW) across all Actual-prefix sites per 5-min timestamp,
    then resample to hourly totals. Returns a DatetimeIndex Series.
    """
    df = pd.read_csv(features_path, parse_dates=[TIME_COL],
                     usecols=[TIME_COL, TARGET_COL, "prefix"])
    df = df[df["prefix"] == "Actual"]
    regional = (df.groupby(TIME_COL)[TARGET_COL]
                  .sum()
                  .resample("h")
                  .sum()
                  .rename("total_power_mw"))
    return regional


def chronological_split(
    series: pd.Series, test_size: float = 0.2
) -> tuple[pd.Series, pd.Series]:
    n       = len(series)
    cutoff  = int(n * (1 - test_size))
    return series.iloc[:cutoff], series.iloc[cutoff:]


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple | None,
    n_splits: int = 5,
) -> tuple[float, float]:
    """
    Manual TimeSeriesSplit cross-validation for ARIMA/SARIMA.
    Each fold fits on an expanding window and predicts the next fold.
    Returns (mean_rmse, std_rmse).
    """
    n      = len(series)
    fold_size = n // (n_splits + 1)
    rmses  = []

    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        val_end   = min(train_end + fold_size, n)

        train_s = series.iloc[:train_end]
        val_s   = series.iloc[train_end:val_end]
        h       = len(val_s)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if seasonal_order:
                fit = SARIMAX(train_s, order=order,
                              seasonal_order=seasonal_order).fit(disp=False)
            else:
                fit = ARIMA(train_s, order=order).fit()

        pred = fit.forecast(steps=h)
        rmses.append(np.sqrt(np.mean((val_s.values - pred.values) ** 2)))

    return float(np.mean(rmses)), float(np.std(rmses))


# ── Persistence helper ────────────────────────────────────────────────────────

def persistence_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Lag-1: predict each hour = same hour 24 hours earlier (daily persistence)."""
    # Use last 24 values of train and tile to fill h steps
    last_day = train.values[-24:]
    reps = (h // 24) + 1
    return np.tile(last_day, reps)[:h]


# ── TimesGPT via Nixtla ───────────────────────────────────────────────────────

def run_timesgpt(
    train: pd.Series,
    test: pd.Series,
) -> dict | None:
    """
    Call Nixtla TimesGPT (zero-shot, no fine-tuning).
    Requires NIXTLA_API_KEY environment variable.
    """
    api_key = os.environ.get("NIXTLA_API_KEY", "")
    if not api_key:
        print("  [skip] NIXTLA_API_KEY not set — skipping TimesGPT.")
        return None

    try:
        from nixtla import NixtlaClient
    except ImportError:
        print("  [skip] nixtla not installed — skipping TimesGPT.")
        return None

    client = NixtlaClient(api_key=api_key)

    # Nixtla expects a DataFrame with 'ds' (datetime) and 'y' (target)
    df_train = pd.DataFrame({"ds": train.index, "y": train.values})
    h        = len(test)

    t0 = time.time()
    fcst = client.forecast(df=df_train, h=h, freq="h", target_col="y")
    elapsed = time.time() - t0

    y_pred = fcst["TimeGPT"].values
    m = metrics(test.values, y_pred)
    m["elapsed"] = elapsed
    return m


# ── Comparison table ──────────────────────────────────────────────────────────

def print_table(rows: list[dict]) -> None:
    print(f"\n{'─' * 84}")
    print(
        f"  {'Model':<28}  {'CV RMSE':>10}  {'CV std':>8}  "
        f"{'Test MAE':>9}  {'Test RMSE':>10}  {'Test R2':>8}  {'Time (s)':>8}"
    )
    print(f"  {'─'*28}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*10}  {'─'*8}  {'─'*8}")
    for r in rows:
        cv_mean = f"{r['CV RMSE']:>10.2f}" if not np.isnan(r['CV RMSE']) else f"{'—':>10}"
        cv_std  = f"{r['CV std']:>8.2f}"  if not np.isnan(r['CV std'])  else f"{'—':>8}"
        t_str   = f"{r['Time (s)']:>8.1f}" if not np.isnan(r['Time (s)']) else f"{'—':>8}"
        print(
            f"  {r['Model']:<28}  {cv_mean}  {cv_std}  "
            f"{r['Test MAE']:>9.2f}  {r['Test RMSE']:>10.2f}  {r['Test R2']:>8.4f}  {t_str}"
        )
    print(f"{'─' * 84}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n_cv_splits: int = 5) -> None:
    t_total = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    NAN = float("nan")

    # ── Build series ──────────────────────────────────────────────────────────
    print("Building regional hourly time series (sum across all Actual sites) ...")
    series = build_regional_series(FEATURES_PATH)
    print(f"  {len(series)} hourly observations")
    print(f"  Range : {series.index[0]}  →  {series.index[-1]}")
    print(f"  Total power  min={series.min():.1f}  max={series.max():.1f}  "
          f"mean={series.mean():.1f} MW")

    train, test = chronological_split(series, test_size=0.2)
    print("\nSplit:")
    print(f"  Train ({len(train):,} hrs): {train.index[0]} → {train.index[-1]}")
    print(f"  Test  ({len(test):,} hrs) : {test.index[0]} → {test.index[-1]}")

    rows = []

    # ── Persistence (daily lag-24) ─────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Persistence (lag-24, same hour yesterday):")
    y_persist = persistence_forecast(train, len(test))
    pm = metrics(test.values, y_persist)
    print(f"  MAE={pm['MAE']:.2f}  RMSE={pm['RMSE']:.2f}  R2={pm['R2']:.4f}")
    rows.append({"Model": "Persistence (lag-24)",
                 "CV RMSE": NAN, "CV std": NAN,
                 "Test MAE": pm["MAE"], "Test RMSE": pm["RMSE"],
                 "Test R2": pm["R2"], "Time (s)": 0.0})

    # ── ARIMA(2,1,2) ──────────────────────────────────────────────────────────
    # WHY (2,1,2): Solar generation has short-term autocorrelation (2 lag terms)
    # and ramp dynamics (2 MA terms). d=1 differencing removes the non-stationarity
    # from daytime/nighttime level shifts. No seasonal term — nighttime zeros are
    # treated as part of the signal rather than explicitly modelled.
    print(f"\n{'─' * 60}")
    print("ARIMA(2,1,2) ...")
    arima_order = (2, 1, 2)

    print(f"  Running {n_cv_splits}-fold walk-forward CV ...")
    t0 = time.time()
    arima_cv_mean, arima_cv_std = walk_forward_cv(train, arima_order, None, n_cv_splits)
    print(f"  CV RMSE: {arima_cv_mean:.2f} ± {arima_cv_std:.2f}  ({time.time()-t0:.1f}s)")

    print("  Fitting on full training set ...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima_fit = ARIMA(train, order=arima_order).fit()
    arima_train_time = time.time() - t0

    arima_pred = arima_fit.forecast(steps=len(test))
    am = metrics(test.values, arima_pred.values)
    print(f"  Test  MAE={am['MAE']:.2f}  RMSE={am['RMSE']:.2f}  R2={am['R2']:.4f}  "
          f"({arima_train_time:.1f}s)")

    joblib.dump(arima_fit, MODELS_DIR / "arima.pkl")
    rows.append({"Model": "ARIMA(2,1,2)",
                 "CV RMSE": arima_cv_mean, "CV std": arima_cv_std,
                 "Test MAE": am["MAE"], "Test RMSE": am["RMSE"],
                 "Test R2": am["R2"], "Time (s)": arima_train_time})

    # ── SARIMA(1,1,1)(1,1,1,24) ───────────────────────────────────────────────
    # WHY: Solar generation has a strong 24-hour seasonal cycle — output at
    # 10:00 today is closely related to output at 10:00 yesterday. SARIMA adds
    # seasonal AR(1), differencing (D=1) and MA(1) with period S=24 to capture
    # this diurnal pattern that ARIMA misses. The seasonal differencing also
    # removes the day-length trend across seasons.
    print(f"\n{'─' * 60}")
    print("SARIMA(1,1,1)(1,1,1,24) ...")
    sarima_order    = (1, 1, 1)
    sarima_seasonal = (1, 1, 1, 24)

    print(f"  Running {n_cv_splits}-fold walk-forward CV ...")
    t0 = time.time()
    sarima_cv_mean, sarima_cv_std = walk_forward_cv(
        train, sarima_order, sarima_seasonal, n_cv_splits
    )
    print(f"  CV RMSE: {sarima_cv_mean:.2f} ± {sarima_cv_std:.2f}  ({time.time()-t0:.1f}s)")

    print("  Fitting on full training set ...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sarima_fit = SARIMAX(train, order=sarima_order,
                             seasonal_order=sarima_seasonal).fit(disp=False)
    sarima_train_time = time.time() - t0

    sarima_pred = sarima_fit.forecast(steps=len(test))
    sm = metrics(test.values, sarima_pred.values)
    print(f"  Test  MAE={sm['MAE']:.2f}  RMSE={sm['RMSE']:.2f}  R2={sm['R2']:.4f}  "
          f"({sarima_train_time:.1f}s)")

    joblib.dump(sarima_fit, MODELS_DIR / "sarima.pkl")
    rows.append({"Model": "SARIMA(1,1,1)(1,1,1,24)",
                 "CV RMSE": sarima_cv_mean, "CV std": sarima_cv_std,
                 "Test MAE": sm["MAE"], "Test RMSE": sm["RMSE"],
                 "Test R2": sm["R2"], "Time (s)": sarima_train_time})

    # ── TimesFM ───────────────────────────────────────────────────────────────
    # WHY: Zero-shot foundation model pre-trained on a large corpus of time
    # series. No fine-tuning required; useful as an out-of-the-box benchmark.
    # SKIPPED: The JAX-based `lingvo` dependency (required by timesfm==1.0.0)
    # has no macOS ARM (arm64) wheel. To run on Linux or in Colab:
    #   pip install timesfm
    #   from timesfm import TimesFm
    #   tfm = TimesFm(context_len=512, horizon_len=len(test), backend="cpu")
    #   tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    #   preds, _ = tfm.forecast(inputs=[train.values], freq=[0])
    print(f"\n{'─' * 60}")
    print("TimesFM: SKIPPED")
    print("  Reason: timesfm==1.0.0 requires JAX `lingvo` package which has")
    print("  no macOS ARM wheel. Re-run on Linux/Colab after `pip install timesfm`.")
    rows.append({"Model": "TimesFM (skipped)",
                 "CV RMSE": NAN, "CV std": NAN,
                 "Test MAE": NAN, "Test RMSE": NAN,
                 "Test R2": NAN, "Time (s)": NAN})

    # ── TimesGPT (Nixtla) ─────────────────────────────────────────────────────
    # WHY: Pre-trained foundation model for zero-shot time series forecasting.
    # Trained on billions of time series data points. No fitting required;
    # sends the training context to Nixtla's API and returns forecasts.
    # Useful as a state-of-the-art zero-shot benchmark.
    print(f"\n{'─' * 60}")
    print("TimesGPT (Nixtla) ...")
    tgpt_result = run_timesgpt(train, test)
    if tgpt_result:
        m = tgpt_result
        print(f"  Test  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  "
              f"({m['elapsed']:.1f}s)")
        rows.append({"Model": "TimesGPT (zero-shot)",
                     "CV RMSE": NAN, "CV std": NAN,
                     "Test MAE": m["MAE"], "Test RMSE": m["RMSE"],
                     "Test R2": m["R2"], "Time (s)": m["elapsed"]})
    else:
        rows.append({"Model": "TimesGPT (no API key)",
                     "CV RMSE": NAN, "CV std": NAN,
                     "Test MAE": NAN, "Test RMSE": NAN,
                     "Test R2": NAN, "Time (s)": NAN})

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'═' * 84}")
    print("COMPARISON TABLE  —  Regional hourly solar generation (MW total)")
    print("(ARIMA/SARIMA operate on the aggregated regional series; ML models")
    print(" in compare_models.py operate per-site — metrics are not directly comparable)\n")
    print_table(rows)

    # ── Best model ────────────────────────────────────────────────────────────
    scored = [r for r in rows if not np.isnan(r["Test RMSE"])]
    if scored:
        best = min(scored, key=lambda r: r["Test RMSE"])
        print(f"\nBest on test RMSE: {best['Model']}  (RMSE={best['Test RMSE']:.2f} MW)")
        pers = next(r for r in rows if "Persistence" in r["Model"])
        gain = (pers["Test RMSE"] - best["Test RMSE"]) / pers["Test RMSE"] * 100
        print(f"  Beats lag-24 persistence by {gain:+.1f}% RMSE")

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare classical and foundation TS models"
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    args = parser.parse_args()
    run(n_cv_splits=args.cv_splits)
