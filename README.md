# Solar PV Generation Forecasting

> End-to-end machine learning pipeline for multi-site, multi-step-ahead solar photovoltaic power forecasting — from raw 5-minute CSVs to a production-ready Streamlit dashboard.

**Live Demo:** [https://your-app-url.streamlit.app](https://your-app-url.streamlit.app) *(deploy link — coming soon)*
&nbsp;|&nbsp;
**GitHub:** [github.com/nsnatthapat/my-ml-project](https://github.com/nsnatthapat/my-ml-project)

[![CI](https://github.com/nsnatthapat/my-ml-project/actions/workflows/ci.yml/badge.svg)](https://github.com/nsnatthapat/my-ml-project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-brightgreen)
![Tests](https://img.shields.io/badge/tests-47%20passing-success)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Results](#3-results)
4. [Tech Stack](#4-tech-stack)
5. [Setup & Installation](#5-setup--installation)
6. [How to Run](#6-how-to-run)
7. [Feature Engineering](#7-feature-engineering)
8. [Key Decisions & Lessons](#8-key-decisions--lessons)
9. [File Structure](#9-file-structure)

---

## 1. Project Overview

### The Problem

Grid operators and energy traders need to know how much solar power each site will generate in the next 5–60 minutes. Getting this wrong means either curtailing clean energy or paying for expensive peaker plants to fill the gap. Accurate short-term forecasting is worth millions per year to a grid operator.

### End User

Energy dispatch teams and grid balancing operators who need site-level power forecasts on a 5-minute cadence to schedule generation assets and balance supply against real-time demand.

### The Data

| Property | Detail |
|---|---|
| Source | NREL Solar Power Data for Integration Studies (Washington State) |
| Sites | 76 photovoltaic sites — 48 distributed (DPV, 0.2–5 MW) + 28 utility-scale (UPV, 1–109 MW) |
| Resolution | 5-minute intervals, full year 2006 |
| Volume | **228 CSV files → 7,989,120 rows** after concatenation |
| File format | `Actual_{lat}_{lon}_{year}_{type}_{capacity}_5_Min.csv` |

### What the Model Outputs

Given a site's recent power history and time-of-day/season context, the model predicts **Power(MW)** at the next 5-minute interval. This is a **regression** problem, not classification — the output is a continuous MW value between 0 and the site's nameplate capacity.

### Key Design Decisions

- **Chronological train/test split** (never shuffle): train on Jan–Oct 2006, evaluate on Oct–Dec 2006. Shuffling would leak future data and inflate all metrics.
- **Per-site lag features** computed via `groupby(["lat", "lon"])` to prevent cross-site contamination — lag-1 of site A must never pick up site B's last reading.
- **Gradient boosting over deep learning**: LightGBM handles the 8M-row tabular dataset efficiently, captures non-linear interactions without feature scaling, and interprets naturally. A transformer would require far more data engineering for marginal gain.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA LAYER                               │
│  228 CSV files · 76 sites · 5-min intervals · full year 2006        │
│  Filename encodes: lat · lon · year · pv_type · capacity            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  src/data/loader.py
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      QUALITY GATE                                   │
│  src/data/quality.py — 5 automated checks:                          │
│  schema · row count · null rates · value ranges · target dist.      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  src/data/cleaner.py
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CLEANING PIPELINE                              │
│  Drop cols >50% null · ffill/bfill · dedup · dtype coercion         │
│  Output: data/cleaned.csv                                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  src/features/engineering.py
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING (27 features)                │
│  Temporal: hour · month · season · sin/cos cyclical encodings       │
│  Site:     capacity_mw · is_utility_scale · lat_normalized          │
│  Composite: daytime_capacity · solar_exposure_index                 │
│  Lag/Roll:  power_lag_{1,2,4,12,24} · rolling_std_12 · mean_288    │
│                                                                     │
│  → select_features(): correlation filter + variance filter          │
│  → 12 features survive into data/features.csv                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
             ┌─────────────┴──────────────┐
             │                            │
             ▼                            ▼
┌────────────────────────┐   ┌────────────────────────────────────────┐
│   MODEL COMPARISON     │   │        CLASSICAL TS MODELS             │
│  src/models/           │   │  src/models/compare_ts_models.py       │
│  compare_models.py     │   │                                        │
│                        │   │  • Regional hourly aggregation (8,760) │
│  • Persistence (lag-1) │   │  • ARIMA(2,1,2)  → R²= −0.19 (FAIL)  │
│  • Linear Regression   │   │  • SARIMA(1,1,1)(1,1,1,24) → R²=0.52 │
│  • Ridge Regression    │   │  • TimesGPT (nixtla API, optional)     │
│  • XGBoost             │   └────────────────────────────────────────┘
│  • LightGBM ← WINNER   │
└────────────┬───────────┘
             │  src/models/tuning.py
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETER TUNING                            │
│  Optuna TPE sampler · 30 trials · 5-fold TimeSeriesSplit CV         │
│  Best: lr=0.0208 · leaves=21 · min_split_gain=0.88                 │
│  Output: models/best_params.json                                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  src/models/run_training.py
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MLFLOW EXPERIMENT TRACKING                       │
│  SQLite backend (mlflow.db) · logs params, metrics, artifacts       │
│  Runs: baseline config + tuned_best config                          │
│  Output: models/production_model.pkl                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                              │
│  app/streamlit_app.py · 4 pages                                     │
│  Project Overview · EDA Explorer · Model Results · How I Built This │
│  Runs in Docker · falls back to demo data if artifacts missing      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Results

All metrics evaluated on the **chronological 20% holdout** (Oct 20 – Dec 31, 2006). No data from this window was used during training or hyperparameter search.

### Tabular Models (site-level, 5-min resolution)

| Model | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ | vs Persistence RMSE |
|---|---|---|---|---|
| Persistence (lag-1) — *naive baseline* | 0.1755 | 1.1330 | 0.9633 | — |
| Linear Regression | 0.2293 | 1.1315 | 0.9634 | −0.1% |
| Ridge Regression | 0.2293 | 1.1315 | 0.9634 | −0.1% |
| XGBoost | 0.1906 | 1.1044 | 0.9652 | −2.5% |
| LightGBM (baseline params) | 0.1858 | 1.0969 | 0.9656 | −3.2% |
| **LightGBM (Optuna-tuned) ★** | **0.1808** | **1.0971** | **0.9656** | **−3.2%** |

### Classical TS Models (regional aggregate, hourly)

| Model | Test RMSE | Test R² | Notes |
|---|---|---|---|
| Persistence (lag-24) | 1,712 MW | 0.496 | Same hour yesterday |
| ARIMA(2,1,2) | 2,524 MW | −0.190 | Failed — no seasonality |
| SARIMA(1,1,1)(1,1,1,24) | 1,601 MW | 0.523 | Beats persistence by +6.7% |

### Why LightGBM Wins

Persistence is already a strong baseline (R² = 0.963) because solar output is highly autocorrelated — lag-1 alone explains 36% of variance. LightGBM adds value by capturing **non-linear interactions** that persistence cannot:

- The relationship between `hour × capacity_mw` is multiplicative and site-specific
- Cloud-cover events show up in `rolling_std_12` — high variability predicts drops
- `min_split_gain = 0.88` (found by Optuna) aggressively prunes weak splits, preventing the model from overfitting the 57% nighttime zeros

---

## 4. Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| **Data** | pandas, NumPy | Ingestion, 8M-row manipulation, feature computation |
| **ML** | LightGBM | Production model — gradient boosting, leaf-wise growth |
| **ML** | XGBoost | Comparison model — depth-wise boosting |
| **ML** | scikit-learn | Ridge, LinearRegression, pipelines, TimeSeriesSplit CV |
| **TS Models** | statsmodels | ARIMA / SARIMA on regional hourly aggregate |
| **Tuning** | Optuna (TPE) | Bayesian hyperparameter search, 30 trials |
| **Tracking** | MLflow | Experiment tracking, metric logging, artifact storage |
| **Tracking** | SQLite | MLflow backend store (`mlflow.db`) |
| **Persistence** | joblib | Model serialisation (`.pkl` artifacts) |
| **Viz** | Plotly | Interactive charts in Streamlit |
| **App** | Streamlit | 4-page portfolio dashboard with live predictions |
| **Containerisation** | Docker + Compose | Reproducible deployment, volume-mounted artifacts |
| **Testing** | pytest | 47 tests across data quality, features, and model |
| **Linting** | ruff | Fast Python linter, enforced in CI |
| **CI/CD** | GitHub Actions | Automated test + lint on every push and PR |

---

## 5. Setup & Installation

### Prerequisites

- Python 3.9+
- Git

### Clone and create environment

```bash
git clone https://github.com/nsnatthapat/my-ml-project.git
cd my-ml-project

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Verify installation

```bash
python -c "import lightgbm, optuna, mlflow, streamlit; print('All dependencies OK')"
pytest tests/ -v                   # 47 tests should pass
```

---

## 6. How to Run

The pipeline has five stages. Run them in order for a full end-to-end reproduction.

### Stage 1 — Feature engineering

```bash
python src/features/run_features.py
# Reads:  data/cleaned.csv  (228 CSVs already pre-cleaned)
# Writes: data/features.csv  (~8M rows, 12 selected features)
# Time:   ~3-5 minutes
```

### Stage 2 — Model comparison (optional, informational)

```bash
python src/models/compare_models.py
# Trains Ridge, XGBoost, LightGBM; prints comparison table
# Time: ~10 minutes (500k-row CV subsample)
```

### Stage 3 — Hyperparameter tuning

```bash
python src/models/tuning.py                  # 30 trials (default)
python src/models/tuning.py --trials 50      # more thorough search
# Writes: models/best_params.json
# Time:   ~20-40 minutes depending on --trials
```

### Stage 4 — MLflow training run

```bash
python src/models/run_training.py
# Trains baseline + tuned configs, logs to mlflow.db
# Writes: models/production_model.pkl
# Time:   ~5-10 minutes
```

### Stage 5 — View MLflow UI

```bash
# In a separate terminal (keep running):
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --host 127.0.0.1 --port 5000

# Then open:  http://127.0.0.1:5000
```

---

### Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
# Open: http://127.0.0.1:8501
```

> The app works without training — if `data/model_results.json` or
> `models/production_model.pkl` are missing it falls back to synthetic
> demo data automatically.

---

### Docker

```bash
# Build and run
docker compose up --build

# Run detached
docker compose up -d

# View logs
docker compose logs -f streamlit

# Open: http://localhost:8501
```

`data/` and `models/` are mounted as volumes — retrain locally and the container picks up new artifacts without a rebuild.

---

### Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a single file
pytest tests/test_features.py -v

# Run a specific test class
pytest tests/test_model.py::TestPredictions -v
```

`test_model.py` is automatically skipped if `models/production_model.pkl` is not present.

---

## 7. Feature Engineering

27 features were engineered across four categories. After applying a **Pearson correlation filter** (|r| > 0.95 drops the second feature) and a **variance filter** (< 1% of mean variance is dropped), 12 features survived into the final model.

| Feature | Category | Importance | Rationale |
|---|---|---|---|
| `power_lag_1` | Lag | **35.9%** | Solar output is highly autocorrelated; previous 5-min reading is the single strongest predictor |
| `rolling_std_12` | Rolling | **15.6%** | High short-term variability signals cloud cover — key for detecting ramp events |
| `hour` | Temporal | **14.7%** | Intraday solar arc; output is near-zero outside 06:00–20:00 at Washington latitudes |
| `power_lag_12` | Lag | **8.2%** | Reading from 1 hour ago — captures longer momentum, smooths transient spikes |
| `month` | Temporal | **7.5%** | Seasonal day-length variation; Pacific NW sees ~5-hour swing between winter and summer |
| `power_lag_24` | Lag | **5.4%** | Reading from 2 hours ago; adds a second anchor for trend direction |
| `rolling_mean_288` | Rolling | **4.9%** | 24-hour rolling average — approximates "same time yesterday" without explicit calendar lag |
| `capacity_mw` | Site | **3.5%** | Nameplate ceiling; UPV sites (1–109 MW) dwarf DPV (0.2–5 MW) in absolute output |
| `solar_exposure_index` | Composite | **2.5%** | `capacity × hour_cos × lat_penalty` — encodes solar geometry and site size jointly |
| `lon` | Site | **1.0%** | East-west position within the study region; eastern Washington gets more sun |
| `daytime_capacity` | Composite | **0.9%** | `is_daytime × capacity_mw` — directly encodes that nameplate only matters when the sun is up |
| `season` | Temporal | **0.1%** | Coarser seasonal bucket; low marginal value given month is already present |

### Features dropped and why

| Feature | Reason dropped |
|---|---|
| `hour_sin`, `hour_cos` | Correlation > 0.95 with `hour` after tree encoding |
| `lat_normalized` | Near-zero variance — only 6 unique latitudes in study region |
| `is_daytime` | Correlation > 0.95 with `hour` (binary threshold of the same signal) |
| `capacity_tier` | Correlation > 0.95 with `capacity_mw` |
| `is_utility_scale` | Correlation > 0.95 with `capacity_mw` (UPV ≡ large capacity) |
| `capacity_normalized_power` | **Target-derived** — uses actual Power(MW) at inference time; would leak the label |

---

## 8. Key Decisions & Lessons

**Chronological split is non-negotiable for time series.**
A random 80/20 shuffle lets the model see October data during training while "predicting" June — inflating R² by several points and producing a model that fails in deployment. Every split, CV fold, and subsample in this project preserves temporal order. The `TimeSeriesSplit` in scikit-learn enforces this inside cross-validation too.

**Per-site lag computation prevents silent data leakage.**
The dataset is a long-format panel with many sites stacked in one DataFrame. A naive `df["Power"].shift(1)` assigns site B's last reading as site A's lag-1 wherever the two sites sit adjacently after sorting. The fix is `groupby(["lat", "lon"])["Power"].shift(n)` — a one-liner, but the bug it prevents would be invisible in aggregate metrics and catastrophic in production when the previous timestep genuinely isn't available.

**ARIMA failed completely — and that's a useful result.**
ARIMA(2,1,2) on the regional aggregate produced R² = −0.19, worse than predicting the mean. The differencing operator removes trend but cannot capture the strong 24-hour seasonality inherent to solar generation. SARIMA(×24) fixed this and beat persistence by 6.7%. The lesson: solar forecasting specifically requires explicit seasonality terms, and testing a model that fails teaches as much as one that succeeds.

**LightGBM's `min_split_gain` was the most impactful tuned parameter.**
Optuna settled on `min_split_gain = 0.88`, aggressively pruning any split that doesn't reduce loss by at least that threshold. Solar data has a structural problem: 57% of rows are exactly zero (nighttime), and shallow splits on zero-heavy leaves appear profitable but don't generalise to the daytime signal. Pruning them cut overfitting without touching daytime regression quality.

**The persistence baseline is harder to beat than it looks.**
With lag-1 at 36% feature importance and persistence already at R² = 0.963, the gain from a full ML pipeline is a 3.2% RMSE reduction. This is humbling but realistic — operational solar forecasting literature routinely reports 2–5% gains over persistence at 5-minute horizons. The real value of the model lies in reducing large-error outliers during cloud-transition events, visible in the residual distribution, not in headline metric improvement.

---

## 9. File Structure

```
my-ml-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions: test + lint on push / PR
│
├── app/
│   └── streamlit_app.py        # 4-page portfolio dashboard
│
├── data/                       # gitignored — generated locally
│   ├── cleaned.csv             # Output of cleaner.py (~8M rows)
│   ├── features.csv            # Output of run_features.py (12 features)
│   ├── predictions.csv         # Test-set predictions from all models
│   └── model_results.json      # Metrics, feature importances, summary stats
│
├── models/                     # gitignored — generated by training
│   ├── best_params.json        # Optuna best trial (all 30 trials logged)
│   ├── production_model.pkl    # Tuned LightGBM — used by Streamlit
│   ├── lightgbm.pkl            # Baseline LightGBM
│   ├── xgboost.pkl             # XGBoost comparison model
│   ├── ridge.pkl               # Ridge Regression comparison model
│   ├── baseline.pkl            # Linear Regression baseline
│   ├── arima.pkl               # ARIMA classical TS model
│   └── sarima.pkl              # SARIMA classical TS model
│
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis (7 sections)
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Parse 228 CSVs, extract metadata from filenames
│   │   ├── quality.py          # 5-check automated data quality gate
│   │   └── cleaner.py          # Dedup, ffill/bfill, dtype coercion → cleaned.csv
│   │
│   ├── features/
│   │   ├── engineering.py      # create_features() — 27 engineered features
│   │   └── run_features.py     # End-to-end pipeline: cleaned.csv → features.csv
│   │
│   └── models/
│       ├── baseline.py         # Persistence + LinearRegression baselines
│       ├── compare_models.py   # Ridge / XGBoost / LightGBM CV comparison
│       ├── compare_ts_models.py# ARIMA / SARIMA / TimesGPT on hourly aggregate
│       ├── tuning.py           # Optuna hyperparameter search (30 trials, TPE)
│       └── run_training.py     # MLflow-tracked final training pipeline
│
├── tests/
│   ├── test_data_quality.py    # 12 tests — quality gate pass/fail behaviour
│   ├── test_features.py        # 20 tests — column count, NaN, value ranges
│   └── test_model.py           # 15 tests — load, predict, physical bounds
│
├── Dockerfile                  # python:3.9-slim, installs deps, exposes 8501
├── docker-compose.yml          # Streamlit service with data/ models/ volumes
├── requirements.txt            # All Python dependencies
├── setup.py                    # Editable install for src/ package
└── README.md                   # This file
```

---

### Reproducing from Scratch

```bash
# 1. Clone
git clone https://github.com/nsnatthapat/my-ml-project.git
cd my-ml-project

# 2. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 3. Full pipeline (assumes raw CSVs are in data/)
python src/features/run_features.py
python src/models/tuning.py
python src/models/run_training.py

# 4. Dashboard
streamlit run app/streamlit_app.py

# 5. Tests
pytest tests/ -v
```

Total pipeline time on a modern laptop: **~30–50 minutes** (dominated by Optuna tuning).

---

*Built with Python · LightGBM · Optuna · MLflow · Streamlit · Docker · pytest · GitHub Actions*
