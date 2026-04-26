"""
Solar PV Generation Forecasting — Portfolio Showcase
Multi-page Streamlit app for hiring manager review.
"""

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"

FEATURES_CSV     = DATA_DIR   / "features.csv"
PREDICTIONS_CSV  = DATA_DIR   / "predictions.csv"
MODEL_RESULTS    = DATA_DIR   / "model_results.json"
PRODUCTION_MODEL = MODELS_DIR / "production_model.pkl"

# ── Color palette ──────────────────────────────────────────────────────────────
SOLAR_ORANGE  = "#F97316"
SOLAR_AMBER   = "#FBBF24"
NAVY          = "#1E3A5F"
TEAL          = "#0D9488"
SLATE         = "#475569"
LIGHT_BG      = "#F8FAFC"
MODEL_COLORS  = {
    "Persistence (lag-1)"  : "#94A3B8",
    "Linear Regression"    : "#64748B",
    "Ridge Regression"     : "#6366F1",
    "XGBoost"              : "#3B82F6",
    "LightGBM (baseline)"  : "#10B981",
    "LightGBM (tuned) ★"  : "#F97316",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar PV Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar nav */
  [data-testid="stSidebar"] { background: #1E3A5F; }
  [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
  [data-testid="stSidebar"] .stRadio label { font-size: 15px; padding: 4px 0; }

  /* KPI cards */
  .kpi-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 5px solid #F97316;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    height: 100%;
  }
  .kpi-label { font-size: 13px; color: #64748B; font-weight: 600;
                text-transform: uppercase; letter-spacing: .05em; }
  .kpi-value { font-size: 32px; font-weight: 700; color: #1E3A5F; margin: 4px 0; }
  .kpi-delta { font-size: 13px; color: #10B981; }

  /* Callout boxes */
  .callout {
    background: #FFF7ED;
    border-left: 4px solid #F97316;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
  }
  .callout-title { font-weight: 700; color: #C2410C; font-size: 14px; }
  .callout-body  { color: #431407; font-size: 14px; margin-top: 4px; }

  /* Badge */
  .badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 600;
    margin: 3px;
  }

  /* Section divider */
  .section-title {
    font-size: 20px;
    font-weight: 700;
    color: #1E3A5F;
    border-bottom: 2px solid #F97316;
    padding-bottom: 6px;
    margin-top: 8px;
    margin-bottom: 16px;
  }

  /* Best row highlight */
  .winner { background-color: #FFF7ED !important; font-weight: 600; }

  /* Footer */
  .footer {
    margin-top: 48px;
    padding: 16px;
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    border-top: 1px solid #E2E8F0;
  }
</style>
""", unsafe_allow_html=True)


# ── Demo-data generators (used when real artifacts don't exist) ────────────────

_FEATURES = [
    "lon", "capacity_mw", "hour", "month", "season",
    "daytime_capacity", "solar_exposure_index",
    "power_lag_1", "power_lag_12", "power_lag_24",
    "rolling_std_12", "rolling_mean_288",
]

def _solar_mw(hour: np.ndarray, month: np.ndarray, capacity: np.ndarray,
              noise_scale: float = 0.0, rng=None) -> np.ndarray:
    """Synthetic solar power: bell curve over daylight hours, winter-attenuated."""
    if rng is None:
        rng = np.random.default_rng(42)
    seasonal = 0.5 + 0.5 * np.cos(np.pi * (month - 7) / 6)   # peak July, low Jan
    daylight  = np.where(
        (hour >= 6) & (hour <= 19),
        np.sin(np.pi * (hour - 6) / 13),
        0.0,
    )
    base = capacity * seasonal * daylight * 0.7
    base = np.clip(base, 0, None)
    if noise_scale > 0:
        base = np.clip(base + rng.normal(0, noise_scale * capacity, size=base.shape), 0, None)
    return base.astype(np.float32)


def _make_demo_results() -> dict:
    rng = np.random.default_rng(42)
    feat_imp_raw = rng.dirichlet([8, 1, 3, 2, 1, 1, 1, 2, 1, 1, 3, 1])
    # Pin the most important ones to realistic values
    feat_imp_raw[_FEATURES.index("power_lag_1")]      = 0.3592
    feat_imp_raw[_FEATURES.index("rolling_std_12")]   = 0.1557
    feat_imp_raw[_FEATURES.index("hour")]              = 0.1469
    feat_imp = feat_imp_raw / feat_imp_raw.sum()

    return {
        "summary": {
            "total_rows"            : 7_989_120,
            "train_rows"            : 6_391_296,
            "test_rows"             : 1_597_824,
            "n_features_engineered" : 27,
            "n_features_selected"   : 12,
            "features"              : _FEATURES,
            "best_model"            : "LightGBM (tuned)",
            "best_rmse"             : 1.0971,
            "best_r2"               : 0.9656,
            "best_mae"              : 0.1808,
            "persist_rmse"          : 1.1330,
            "rmse_improvement_pct"  : 3.17,
            "n_sites"               : 76,
            "date_range"            : ["2006-01-01", "2006-12-31"],
            "zero_fraction"         : 0.5730,
        },
        "model_comparison": [
            {"model": "Persistence (lag-1)",   "MAE": 0.1755, "RMSE": 1.1330, "R2": 0.9633},
            {"model": "Linear Regression",     "MAE": 0.2293, "RMSE": 1.1315, "R2": 0.9634},
            {"model": "Ridge Regression",      "MAE": 0.2293, "RMSE": 1.1315, "R2": 0.9634},
            {"model": "XGBoost",               "MAE": 0.1906, "RMSE": 1.1044, "R2": 0.9652},
            {"model": "LightGBM (baseline)",   "MAE": 0.1858, "RMSE": 1.0969, "R2": 0.9656},
            {"model": "LightGBM (tuned) ★",   "MAE": 0.1808, "RMSE": 1.0971, "R2": 0.9656},
        ],
        "feature_importances": {f: round(float(v), 4) for f, v in zip(_FEATURES, feat_imp)},
        "best_params": {
            "model": "LightGBM", "best_trial": 13, "cv_rmse": 1.31527, "n_trials": 30,
            "cv_splits": 5,
            "hyperparameters": {
                "num_leaves": 21, "max_depth": 6, "min_child_samples": 137,
                "learning_rate": 0.02079, "n_estimators": 505,
                "subsample": 0.6635, "colsample_bytree": 0.9656,
                "reg_alpha": 1.34e-05, "reg_lambda": 5.96e-07, "min_split_gain": 0.8796,
            },
        },
    }


def _make_demo_predictions(n: int = 20_000) -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2006-10-20", "2006-12-31 23:55", freq="5min")
    idx   = rng.choice(len(dates), size=n, replace=False)
    idx.sort()
    times = dates[idx]

    hours    = times.hour.to_numpy().astype(float)
    months   = times.month.to_numpy().astype(float)
    capacity = rng.choice([0.2, 0.3, 0.4, 1.0, 2.0, 4.0, 5.0, 30.0, 50.0], size=n)

    actual       = _solar_mw(hours, months, capacity, noise_scale=0.08, rng=rng)
    persistence  = np.roll(actual, 1)
    persistence[0] = actual[0]
    lgbm_pred    = _solar_mw(hours, months, capacity, noise_scale=0.04, rng=rng)
    xgb_pred     = _solar_mw(hours, months, capacity, noise_scale=0.05, rng=rng)
    ridge_pred   = _solar_mw(hours, months, capacity, noise_scale=0.09, rng=rng)
    tuned_pred   = _solar_mw(hours, months, capacity, noise_scale=0.03, rng=rng)

    return pd.DataFrame({
        "LocalTime"      : times,
        "actual"         : actual,
        "persistence"    : persistence,
        "lightgbm"       : lgbm_pred,
        "xgboost"        : xgb_pred,
        "ridge"          : ridge_pred,
        "tuned_lightgbm" : tuned_pred,
        "hour"           : hours.astype(int),
        "month"          : months.astype(int),
    })


def _make_demo_features(n: int = 200_000) -> pd.DataFrame:
    rng      = np.random.default_rng(42)
    dates    = pd.date_range("2006-01-01", "2006-12-31 23:55", freq="5min")
    idx      = rng.choice(len(dates), size=n, replace=False)
    idx.sort()
    times    = dates[idx]
    hours    = times.hour.to_numpy().astype(float)
    months   = times.month.to_numpy().astype(float)
    capacity = rng.choice([0.2, 0.3, 0.4, 1.0, 2.0, 4.0, 5.0, 30.0, 50.0, 109.0], size=n)
    lon      = rng.choice(np.linspace(-123.0, -117.0, 28), size=n)

    power = _solar_mw(hours, months, capacity, noise_scale=0.10, rng=rng)
    lag1  = _solar_mw(hours - 1/12, months, capacity, noise_scale=0.10, rng=rng)
    lag12 = _solar_mw(hours - 1,    months, capacity, noise_scale=0.12, rng=rng)
    lag24 = _solar_mw(hours - 2,    months, capacity, noise_scale=0.12, rng=rng)

    season = np.where(months.isin([12, 1, 2])  if hasattr(months, 'isin') else
                      np.isin(months, [12, 1, 2]), 1,
             np.where(np.isin(months, [3, 4, 5]), 2,
             np.where(np.isin(months, [6, 7, 8]), 3, 4)))

    return pd.DataFrame({
        "LocalTime"            : times,
        "Power(MW)"            : power,
        "lon"                  : lon,
        "capacity_mw"          : capacity,
        "hour"                 : hours.astype(int),
        "month"                : months.astype(int),
        "season"               : season,
        "daytime_capacity"     : np.where((hours >= 6) & (hours <= 19), capacity, 0),
        "solar_exposure_index" : np.clip(power / (capacity + 1e-6), 0, 1),
        "power_lag_1"          : lag1,
        "power_lag_12"         : lag12,
        "power_lag_24"         : lag24,
        "rolling_std_12"       : np.abs(rng.normal(0.5, 0.3, n)),
        "rolling_mean_288"     : (lag1 + lag12 + lag24) / 3,
        "pv_type"              : rng.choice(["DPV", "UPV"], size=n),
    })


class _DemoModel:
    """Formula-based stand-in when production_model.pkl hasn't been trained yet."""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        h   = X["hour"].to_numpy(dtype=float)
        m   = X["month"].to_numpy(dtype=float)
        cap = X["capacity_mw"].to_numpy(dtype=float)
        lag = X["power_lag_1"].to_numpy(dtype=float)
        base = _solar_mw(h, m, cap, noise_scale=0.0)
        # Blend formula with lag-1 (mirrors what a real model learns)
        return np.clip(0.6 * base + 0.4 * lag, 0, None)


# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_results() -> dict:
    if MODEL_RESULTS.exists():
        with open(MODEL_RESULTS) as f:
            return json.load(f)
    return _make_demo_results()


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_CSV.exists():
        return pd.read_csv(PREDICTIONS_CSV, parse_dates=["LocalTime"]).dropna()
    return _make_demo_predictions()


@st.cache_data(show_spinner=False)
def load_features_sample(n: int = 200_000) -> pd.DataFrame:
    if FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV, parse_dates=["LocalTime"])
        df = df.sort_values("LocalTime").reset_index(drop=True)
        day   = df[df["Power(MW)"] > 0].sample(n=n // 2, random_state=42)
        night = df[df["Power(MW)"] == 0].sample(n=n // 2, random_state=42)
        return pd.concat([day, night]).sort_values("LocalTime").reset_index(drop=True)
    return _make_demo_features(n)


@st.cache_resource(show_spinner=False)
def load_model() -> dict:
    if PRODUCTION_MODEL.exists():
        return joblib.load(PRODUCTION_MODEL)
    return {"model": _DemoModel(), "features": _FEATURES}


def _is_demo() -> bool:
    return not MODEL_RESULTS.exists()


def _demo_banner() -> None:
    if _is_demo():
        st.info(
            "**Demo mode** — training artifacts not found. "
            "All charts use synthetic data that mirrors the real dataset's structure. "
            "Run `python src/models/run_training.py` to load real results.",
            icon="🔬",
        )


# ── Shared header ──────────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        f"""<div style="background: linear-gradient(135deg, {NAVY} 0%, #2D5986 100%);
            padding: 20px 28px; border-radius: 12px; margin-bottom: 24px;
            display: flex; align-items: center; gap: 16px;">
          <div style="font-size: 40px;">☀️</div>
          <div>
            <div style="font-size: 22px; font-weight: 800; color: white;">
              Solar PV Generation Forecasting
            </div>
            <div style="font-size: 14px; color: #93C5FD; margin-top: 2px;">
              Multi-site time-series ML pipeline · Pacific Northwest · 2006
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """<div class="footer">
          Built with Python · LightGBM · Optuna · MLflow · Streamlit &nbsp;|&nbsp;
          <a href="https://github.com/nsnatthapat/my-ml-project" target="_blank"
             style="color:#F97316;">GitHub →</a>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar navigation ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """<div style="padding: 12px 0 20px 0; text-align: center;">
          <div style="font-size: 28px;">☀️</div>
          <div style="font-size: 16px; font-weight: 700; color: #F97316; margin-top: 4px;">
            Solar PV Forecast
          </div>
          <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Portfolio Showcase</div>
        </div>""",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["🏠  Project Overview",
         "📊  Explore the Data",
         "🤖  Model Results",
         "🔧  How I Built This"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """<div style="font-size: 12px; color: #64748B; text-align: center; padding: 8px;">
          <a href="https://github.com/nsnatthapat/my-ml-project" target="_blank"
             style="color:#F97316; text-decoration:none;">📂 GitHub Repo</a>
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if "Overview" in page:
    render_header()
    _demo_banner()
    res = load_results()
    s   = res["summary"]

    # ── What this project does ────────────────────────────────────────────────
    st.markdown('<div class="section-title">What This Project Does</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        This end-to-end machine learning pipeline forecasts solar photovoltaic (PV) power
        generation across **76 sites** in the Pacific Northwest, covering both rooftop
        distributed PV (DPV) and utility-scale UPV installations.
        The pipeline ingests 5-minute resolution actual-generation data, engineers
        temporal and lag features, selects the most predictive signals, tunes a gradient
        boosting model with Optuna, and tracks every experiment with MLflow — all
        production-ready from data loading to model artifact storage.
        The model is evaluated on a **strict chronological holdout** (Oct–Dec 2006),
        never leaking future data into training.
        """
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="kpi-card">
              <div class="kpi-label">Data Points Analyzed</div>
              <div class="kpi-value">{s['total_rows']/1e6:.1f}M</div>
              <div class="kpi-delta">76 sites · 5-min intervals · full year</div>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"""<div class="kpi-card">
              <div class="kpi-label">Features Engineered</div>
              <div class="kpi-value">{s['n_features_engineered']}</div>
              <div class="kpi-delta">{s['n_features_selected']} selected after correlation &amp; variance filter</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"""<div class="kpi-card">
              <div class="kpi-label">Model R² (test set)</div>
              <div class="kpi-value">{s['best_r2']:.4f}</div>
              <div class="kpi-delta">LightGBM tuned · Oct–Dec 2006 holdout</div>
            </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(
            f"""<div class="kpi-card" style="border-left-color: #10B981;">
              <div class="kpi-label">RMSE vs Persistence</div>
              <div class="kpi-value">−{s['rmse_improvement_pct']:.1f}%</div>
              <div class="kpi-delta">{s['persist_rmse']:.4f} → {s['best_rmse']:.4f} MW</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Additional stats row ──────────────────────────────────────────────────
    ca, cb, cc = st.columns(3)
    ca.metric("PV Sites", "76", "28 unique longitudes")
    cb.metric("Zero-generation fraction", f"{s['zero_fraction']*100:.1f}%",
              "nighttime & overcast periods")
    cc.metric("Optuna trials", "30", "TPE sampler · 5-fold TimeSeriesSplit CV")

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Tech Stack</div>', unsafe_allow_html=True)
    stack = {
        "Data": ["pandas", "NumPy", "pathlib"],
        "ML":   ["LightGBM", "XGBoost", "scikit-learn"],
        "Tuning": ["Optuna (TPE)", "TimeSeriesSplit"],
        "Tracking": ["MLflow", "SQLite backend"],
        "TS Models": ["statsmodels ARIMA", "SARIMA"],
        "Viz / App": ["Plotly", "Streamlit"],
    }
    cols = st.columns(len(stack))
    for col, (category, libs) in zip(cols, stack.items()):
        with col:
            st.markdown(f"**{category}**")
            for lib in libs:
                st.markdown(f'<span class="badge">{lib}</span>',
                            unsafe_allow_html=True)

    # ── Pipeline diagram ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Pipeline at a Glance</div>',
                unsafe_allow_html=True)
    st.graphviz_chart("""
        digraph pipeline {
          rankdir=LR;
          graph [bgcolor="transparent" fontname="Arial"]
          node  [shape=box style="rounded,filled" fontname="Arial" fontsize=11
                 fillcolor="#EFF6FF" color="#3B82F6" fontcolor="#1E3A5F"]
          edge  [color="#94A3B8" fontsize=10 fontname="Arial"]

          A [label="228 CSV Files\n(76 sites × 3 types)" fillcolor="#FFF7ED" color="#F97316"]
          B [label="Data Loading\n& Quality Gate"]
          C [label="Cleaning\n& Dedup"]
          D [label="EDA\n(8M rows)"]
          E [label="Feature\nEngineering\n(27 features)"]
          F [label="Feature\nSelection\n(12 features)"]
          G [label="Model\nComparison\n(Ridge / XGB / LGB)"]
          H [label="Optuna\nTuning\n(30 trials)"]
          I [label="MLflow\nTracking"]
          J [label="Production\nModel.pkl" fillcolor="#F0FDF4" color="#10B981"]

          A->B->C->D->E->F->G->H->I->J
        }
    """)

    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE THE DATA
# ══════════════════════════════════════════════════════════════════════════════

elif "Data" in page:
    render_header()
    _demo_banner()
    res = load_results()

    with st.spinner("Loading data sample…"):
        df = load_features_sample(200_000)

    # ── Key findings callouts ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Key EDA Findings</div>',
                unsafe_allow_html=True)
    findings = [
        ("Zero-inflation (57% zeros)",
         "Over half of all 5-minute readings are exactly zero — nighttime, overcast periods, "
         "and winter months. This means the target distribution is heavily left-skewed and a "
         "naive mean prediction would be wildly pessimistic during peak hours."),
        ("power_lag_1 dominates (36% importance)",
         "The single strongest predictor is the previous 5-minute reading. Solar output is "
         "highly autocorrelated at short lags; clouds pass gradually and ramp-up is smooth. "
         "This is why the persistence baseline (ŷ = lag-1) is already R² = 0.963."),
        ("Hour-of-day is the key temporal driver",
         "After lag features, hour of day explains 15% of model importance. The solar arc "
         "creates a sharp bell curve peaking around 13:00 local time. Month adds seasonality "
         "but contributes less than the intraday signal."),
        ("Capacity_mw sets the ceiling, not the mean",
         "Larger installations obviously produce more power — but the relationship is "
         "non-linear. UPV sites (utility-scale, 1–109 MW) exhibit curtailment-like flat "
         "tops during peak hours, visible in daytime_capacity × solar_exposure interaction."),
    ]
    for title, body in findings:
        st.markdown(
            f"""<div class="callout">
              <div class="callout-title">▶ {title}</div>
              <div class="callout-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Target distribution ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Target Distribution — Power (MW)</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        nonzero = df[df["Power(MW)"] > 0]["Power(MW)"]
        fig = px.histogram(
            nonzero,
            nbins=80,
            title="Daytime output (zeros excluded)",
            color_discrete_sequence=[SOLAR_ORANGE],
            labels={"value": "Power (MW)", "count": "Frequency"},
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        hourly_avg = (
            df.groupby("hour")["Power(MW)"].mean().reset_index()
        )
        fig2 = px.area(
            hourly_avg, x="hour", y="Power(MW)",
            title="Average output by hour of day",
            color_discrete_sequence=[SOLAR_AMBER],
            labels={"Power(MW)": "Avg Power (MW)", "hour": "Hour of day"},
        )
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Monthly & seasonal patterns ───────────────────────────────────────────
    st.markdown('<div class="section-title">Seasonal Patterns</div>',
                unsafe_allow_html=True)

    col_m, col_s = st.columns(2)
    with col_m:
        monthly = df.groupby("month")["Power(MW)"].mean().reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda x: month_names[x-1])
        fig3 = px.bar(
            monthly, x="month_name", y="Power(MW)",
            title="Average power by month",
            color="Power(MW)", color_continuous_scale="YlOrRd",
            labels={"Power(MW)": "Avg Power (MW)", "month_name": "Month"},
        )
        fig3.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=20, l=20, r=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_s:
        season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
        df["season_name"] = df["season"].map(season_map)
        season_box = df[df["Power(MW)"] > 0].copy()
        fig4 = px.box(
            season_box,
            x="season_name", y="Power(MW)",
            title="Output distribution by season (daytime only)",
            color="season_name",
            color_discrete_sequence=[NAVY, TEAL, SOLAR_ORANGE, SOLAR_AMBER],
            category_orders={"season_name": ["Spring","Summer","Autumn","Winter"]},
            labels={"Power(MW)": "Power (MW)", "season_name": "Season"},
        )
        fig4.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Correlation Heatmap</div>',
                unsafe_allow_html=True)

    numeric_cols = [
        "Power(MW)", "hour", "month", "season", "capacity_mw",
        "daytime_capacity", "solar_exposure_index",
        "power_lag_1", "power_lag_12", "power_lag_24",
        "rolling_std_12", "rolling_mean_288",
    ]
    corr = df[numeric_cols].corr()
    fig5 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Pearson correlations across features & target",
    )
    fig5.update_layout(
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ── Feature explorer ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Interactive Feature Explorer</div>',
                unsafe_allow_html=True)

    feature_options = [c for c in numeric_cols if c != "Power(MW)"]
    sel_feat = st.selectbox("Select feature to plot against Power (MW):",
                            feature_options, index=feature_options.index("power_lag_1"))

    sample_scatter = df[df["Power(MW)"] > 0].sample(n=min(15_000, len(df)), random_state=7)
    fig6 = px.scatter(
        sample_scatter, x=sel_feat, y="Power(MW)",
        color="hour",
        color_continuous_scale="YlOrRd",
        opacity=0.4,
        trendline="lowess",
        trendline_color_override=NAVY,
        title=f"{sel_feat} vs Power (MW) — coloured by hour of day",
        labels={"Power(MW)": "Power (MW)", sel_feat: sel_feat},
    )
    fig6.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=20, l=20, r=20),
    )
    st.plotly_chart(fig6, use_container_width=True)

    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif "Model" in page:
    render_header()
    _demo_banner()
    res  = load_results()
    preds = load_predictions()

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Comparison (Oct–Dec 2006 holdout)</div>',
                unsafe_allow_html=True)

    models_df = pd.DataFrame(res["model_comparison"])
    models_df["RMSE Improvement vs Persistence"] = (
        (1.1330 - models_df["RMSE"]) / 1.1330 * 100
    ).round(2).astype(str) + "%"
    models_df.loc[models_df["model"] == "Persistence (lag-1)",
                  "RMSE Improvement vs Persistence"] = "—"

    # Highlight best row
    def highlight_winner(row):
        if "tuned" in row["Model"].lower():
            return ["background-color: #FFF7ED; font-weight: 600"] * len(row)
        return [""] * len(row)

    display_df = models_df.rename(columns={
        "model": "Model", "MAE": "MAE ↓", "RMSE": "RMSE ↓", "R2": "R² ↑",
        "RMSE Improvement vs Persistence": "vs Persistence"
    })
    st.dataframe(
        display_df.style.apply(highlight_winner, axis=1).format(
            {"MAE ↓": "{:.4f}", "RMSE ↓": "{:.4f}", "R² ↑": "{:.4f}"}
        ),
        use_container_width=True, hide_index=True,
    )

    # ── Why LightGBM won ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Why LightGBM (tuned) Won</div>',
                unsafe_allow_html=True)

    reasons = [
        ("Non-linear interactions captured",
         "Power output is not additive: the effect of hour-of-day depends on month, "
         "capacity, and lag simultaneously. Tree models split on these interactions "
         "natively — Ridge Regression with linear feature combinations can't."),
        ("Zero-inflation handled implicitly",
         "With 57% zeros in the target, trees learn dedicated subtrees for night/day "
         "regimes. Linear models see all residuals equally; they mis-predict ramp-up "
         "and ramp-down transitions more often."),
        ("Leaf-wise growth + Optuna tuning",
         "LightGBM's leaf-wise strategy explores deep asymmetric splits that capture "
         "outlier peaks. Optuna's 30-trial TPE search found min_split_gain=0.88, "
         "aggressively pruning weak splits — this reduced overfitting without sacrificing "
         "the meaningful non-linear structure."),
        ("Lag features pair naturally with trees",
         "power_lag_1 (36% importance) creates an almost step-function relationship "
         "with the target. Trees use binary threshold splits, which perfectly model "
         "that step structure; polynomial/linear models smooth it out incorrectly."),
    ]
    col_a, col_b = st.columns(2)
    for i, (title, body) in enumerate(reasons):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            st.markdown(
                f"""<div class="callout">
                  <div class="callout-title">✓ {title}</div>
                  <div class="callout-body">{body}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Feature Importances — Production Model</div>',
                unsafe_allow_html=True)

    fi = res["feature_importances"]
    fi_df = pd.DataFrame(
        sorted(fi.items(), key=lambda x: x[1]),
        columns=["Feature", "Importance"],
    )
    fi_df["pct"] = (fi_df["Importance"] * 100).round(1)

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h",
        text="pct",
        color="Importance",
        color_continuous_scale=["#BFDBFE", SOLAR_ORANGE],
        labels={"Importance": "Relative importance", "Feature": ""},
    )
    fig_fi.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_fi.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20, l=20, r=60),
        coloraxis_showscale=False,
        height=380,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")

    # ── Residual plot ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Residual Analysis — LightGBM (tuned)</div>',
                unsafe_allow_html=True)

    preds["residual"] = preds["actual"] - preds["tuned_lightgbm"]
    preds["residual_pct_err"] = (
        np.where(preds["actual"] > 0,
                 preds["residual"] / preds["actual"] * 100,
                 np.nan)
    )

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        sample_r = preds.sample(n=min(8_000, len(preds)), random_state=1)
        fig_res = px.scatter(
            sample_r, x="tuned_lightgbm", y="residual",
            opacity=0.35,
            color_discrete_sequence=[TEAL],
            title="Residuals vs Predicted (MW)",
            labels={"tuned_lightgbm": "Predicted (MW)", "residual": "Residual (actual − pred)"},
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color=SLATE)
        fig_res.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_res2:
        fig_rh = px.histogram(
            preds, x="residual",
            nbins=80,
            color_discrete_sequence=[NAVY],
            title="Distribution of residuals",
            labels={"residual": "Residual (MW)", "count": "Frequency"},
        )
        fig_rh.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_rh, use_container_width=True)

    # ── Actual vs predicted time series ──────────────────────────────────────
    st.markdown('<div class="section-title">Actual vs Predicted — Test Period Sample</div>',
                unsafe_allow_html=True)

    day_preds = preds[preds["actual"] > 0].copy()
    window = day_preds.sort_values("LocalTime").head(2000)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=window["LocalTime"], y=window["actual"],
        mode="lines", name="Actual",
        line=dict(color=NAVY, width=1.5),
    ))
    fig_ts.add_trace(go.Scatter(
        x=window["LocalTime"], y=window["tuned_lightgbm"],
        mode="lines", name="LightGBM (tuned)",
        line=dict(color=SOLAR_ORANGE, width=1.5, dash="dot"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=window["LocalTime"], y=window["persistence"],
        mode="lines", name="Persistence",
        line=dict(color="#94A3B8", width=1, dash="dash"),
    ))
    fig_ts.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title="Date", yaxis_title="Power (MW)",
        height=320,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # ── Try it yourself ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Try It Yourself — Live Prediction</div>',
                unsafe_allow_html=True)
    st.caption("Adjust the sliders to simulate site conditions and get an instant model prediction.")

    art = load_model()
    prod_model    = art["model"]
    prod_features = art["features"]

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        hour       = st.slider("Hour of day", 0, 23, 13)
        month      = st.slider("Month", 1, 12, 7)
        season     = st.selectbox("Season", [1, 2, 3, 4],
                                  format_func=lambda x: {1:"Winter",2:"Spring",3:"Summer",4:"Autumn"}[x],
                                  index=2)
    with col_in2:
        capacity   = st.slider("Site capacity (MW)", 0.1, 109.0, 5.0, step=0.1)
        lon        = st.slider("Longitude", -123.0, -117.0, -120.0, step=0.05)
        lag1       = st.slider("Power lag-1 (prev 5 min, MW)", 0.0, 103.0, 3.0)
    with col_in3:
        lag12      = st.slider("Power lag-12 (prev 1 hr, MW)", 0.0, 103.0, 2.5)
        lag24      = st.slider("Power lag-24 (prev 2 hrs, MW)", 0.0, 103.0, 2.0)
        roll_std   = st.slider("Rolling std-12 (variability)", 0.0, 20.0, 1.0, step=0.1)

    # Derived features
    daytime_cap = capacity if 6 <= hour <= 19 else 0.0
    solar_idx   = max(0.0, capacity * np.sin(np.pi * (hour - 6) / 14)) if 6 <= hour <= 20 else 0.0
    roll_mean   = (lag1 + lag12 + lag24) / 3

    input_row = pd.DataFrame([{
        "lon"                   : lon,
        "capacity_mw"           : capacity,
        "hour"                  : hour,
        "month"                 : month,
        "season"                : season,
        "daytime_capacity"      : daytime_cap,
        "solar_exposure_index"  : round(solar_idx, 4),
        "power_lag_1"           : lag1,
        "power_lag_12"          : lag12,
        "power_lag_24"          : lag24,
        "rolling_std_12"        : roll_std,
        "rolling_mean_288"      : round(roll_mean, 4),
    }])[prod_features]

    prediction = float(np.clip(prod_model.predict(input_row)[0], 0, None))

    col_pred, col_gauge = st.columns([1, 2])
    with col_pred:
        st.metric("Predicted Power Output", f"{prediction:.3f} MW",
                  delta=f"{prediction - lag1:+.3f} vs lag-1")
        capacity_util = prediction / capacity * 100 if capacity > 0 else 0
        st.metric("Capacity Utilization", f"{capacity_util:.1f}%")
        if prediction < 0.01:
            st.info("💤 Model predicts near-zero output (nighttime or low light)")
        elif capacity_util > 60:
            st.success("☀️ High generation — conditions look good!")
        else:
            st.warning("⛅ Moderate generation — partial sun or off-peak hour")

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Predicted MW", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, capacity], "tickwidth": 1},
                "bar": {"color": SOLAR_ORANGE},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, capacity * 0.3], "color": "#FEF3C7"},
                    {"range": [capacity * 0.3, capacity * 0.7], "color": "#FDE68A"},
                    {"range": [capacity * 0.7, capacity], "color": "#FCA5A5"},
                ],
                "threshold": {
                    "line": {"color": NAVY, "width": 3},
                    "thickness": 0.75,
                    "value": capacity,
                },
            },
        ))
        fig_gauge.update_layout(
            margin=dict(t=30, b=10, l=30, r=30),
            paper_bgcolor="white",
            height=220,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HOW I BUILT THIS
# ══════════════════════════════════════════════════════════════════════════════

elif "Built" in page:
    render_header()
    _demo_banner()

    # ── Architecture diagram ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">System Architecture</div>',
                unsafe_allow_html=True)
    st.graphviz_chart("""
        digraph arch {
          rankdir=TB;
          graph [bgcolor="transparent" fontname="Arial" splines=ortho]
          node  [shape=box style="rounded,filled" fontname="Arial" fontsize=11
                 margin="0.3,0.15"]
          edge  [color="#94A3B8" fontname="Arial" fontsize=10]

          subgraph cluster_data {
            label="Data Layer" style=filled fillcolor="#EFF6FF" color="#3B82F6"
            fontname="Arial" fontsize=13
            RAW  [label="228 Raw CSVs\n(5-min, 76 sites)" fillcolor="#DBEAFE" color="#3B82F6"]
            LOAD [label="loader.py\n+ quality.py" fillcolor="#DBEAFE" color="#3B82F6"]
            CLEAN[label="cleaner.py\n→ cleaned.csv"  fillcolor="#DBEAFE" color="#3B82F6"]
          }

          subgraph cluster_features {
            label="Feature Layer" style=filled fillcolor="#F0FDF4" color="#10B981"
            fontname="Arial" fontsize=13
            ENG  [label="engineering.py\n(27 features)" fillcolor="#DCFCE7" color="#10B981"]
            SEL  [label="select_features()\n(12 selected)"fillcolor="#DCFCE7" color="#10B981"]
            FEAT [label="features.csv\n(8M rows)"        fillcolor="#DCFCE7" color="#10B981"]
          }

          subgraph cluster_model {
            label="Model Layer" style=filled fillcolor="#FFF7ED" color="#F97316"
            fontname="Arial" fontsize=13
            COMP [label="compare_models.py\n(Ridge / XGB / LGB)" fillcolor="#FED7AA" color="#F97316"]
            TUNE [label="tuning.py\n(Optuna 30 trials)"          fillcolor="#FED7AA" color="#F97316"]
            MLOG [label="MLflow\n(SQLite tracking)"              fillcolor="#FED7AA" color="#F97316"]
            PROD [label="production_model.pkl" fillcolor="#FED7AA" color="#F97316" shape=cylinder]
          }

          subgraph cluster_serve {
            label="Serving Layer" style=filled fillcolor="#F5F3FF" color="#6366F1"
            fontname="Arial" fontsize=13
            APP  [label="Streamlit App\n(this dashboard)" fillcolor="#DDD6FE" color="#6366F1"]
          }

          RAW->LOAD->CLEAN->ENG->SEL->FEAT
          FEAT->COMP->TUNE->MLOG
          TUNE->PROD->APP
        }
    """)

    # ── Build timeline ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Build Timeline</div>',
                unsafe_allow_html=True)

    timeline = [
        ("Day 1", "Data Foundation",
         "Scaffolded the project, wrote loader.py to parse 228 CSVs with metadata "
         "embedded in filenames, built quality.py for automated data gate checks."),
        ("Day 2", "Cleaning & EDA",
         "Built cleaner.py (dedup, forward-fill, dtype coercion). Generated a full "
         "EDA notebook on 8M rows: discovered 57% zero-inflation and capacity-power "
         "non-linearity."),
        ("Day 3", "Feature Engineering",
         "Engineered 27 features: per-site lag features (groupby lat/lon to prevent "
         "cross-site leakage), rolling statistics, temporal cyclical features, and "
         "solar geometry proxies. Applied correlation + variance filter → 12 features."),
        ("Day 4", "Baseline & Model Comparison",
         "Implemented persistence baseline, LinearRegression, Ridge, XGBoost, and "
         "LightGBM. Used 5-fold TimeSeriesSplit CV on 500k subsample. LightGBM won "
         "with RMSE=1.097 vs persistence 1.133."),
        ("Day 5", "Classical TS Models",
         "Added compare_ts_models.py — aggregated regional hourly series, fit "
         "ARIMA(2,1,2) and SARIMA(1,1,1)(1,1,1,24). ARIMA failed (R²=−0.19); SARIMA "
         "beat persistence by 6.7% on aggregated regional output."),
        ("Day 6", "Hyperparameter Tuning & MLflow",
         "30-trial Optuna TPE search on LightGBM. Best trial #13 (lr=0.021, "
         "leaves=21, min_split_gain=0.88). Integrated MLflow with SQLite backend — "
         "every run logs params, metrics, CV scores, and feature importances."),
        ("Day 7", "Portfolio App",
         "Built this Streamlit dashboard — interactive EDA, model comparison, live "
         "predictions, and architecture documentation. Ready for public deployment."),
    ]

    for day, title, description in timeline:
        col_day, col_body = st.columns([1, 5])
        with col_day:
            st.markdown(
                f"""<div style="background:{NAVY}; color:white; border-radius:8px;
                    padding:10px; text-align:center; font-weight:700; font-size:14px;">
                  {day}
                </div>""", unsafe_allow_html=True)
        with col_body:
            st.markdown(
                f"""<div style="border-left:3px solid {SOLAR_ORANGE}; padding-left:16px;
                    margin-bottom:4px;">
                  <div style="font-weight:700; color:{NAVY}; font-size:15px;">{title}</div>
                  <div style="color:{SLATE}; font-size:14px; margin-top:3px;">{description}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Key decisions ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Key Decisions & Lessons Learned</div>',
                unsafe_allow_html=True)

    decisions = [
        ("Chronological split — not random",
         "Time series data must never be randomly shuffled. Shuffling lets the model "
         "see future data during training, inflating all metrics. I enforce a global "
         "timestamp cutoff: all rows before Oct 20 → train, all after → test."),
        ("Per-site lags to prevent cross-site leakage",
         "Computing lag features with df.groupby(['lat','lon'])['Power'].shift(n) "
         "ensures each site's lag only looks at its own history. A global shift would "
         "corrupt lag-1 with the previous row from a different site at the same timestamp."),
        ("500k CV subsample (not full 8M)",
         "5-fold CV on 8M rows would take ~40 minutes per trial. Subsampling to 500k "
         "contiguous rows (middle slice, preserving temporal structure) cuts this to "
         "~2 minutes while preserving realistic fold boundaries."),
        ("ARIMA failed — here's why",
         "ARIMA on the raw 8M multi-site series is nonsensical. I aggregated to a "
         "regional hourly series (76 sites summed, resampled to 1h → 8,760 points). "
         "Even then, ARIMA(2,1,2) had R²=−0.19 because solar has strong seasonality "
         "that ARIMA's difference operator can't capture — SARIMA(×24) handled it."),
        ("MLflow SQLite vs file store",
         "The default MLflow file store was deprecated in recent versions and produced "
         "FutureWarnings. Switching to sqlite:///mlflow.db gives a proper relational "
         "backend, queryable with SQL, and trivially portable."),
        ("min_split_gain=0.88 — the most important tuned param",
         "Optuna found that aggressively pruning weak splits (min_split_gain=0.88) "
         "was key for this dataset. Solar data has many near-zero rows that create "
         "shallow splits with little gain — pruning them reduces model complexity "
         "without hurting accuracy on the daytime signal."),
    ]

    for title, body in decisions:
        with st.expander(f"💡 {title}"):
            st.markdown(body)

    st.markdown("---")

    # ── GitHub link ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Source Code</div>', unsafe_allow_html=True)
    col_gh, _ = st.columns([1, 2])
    with col_gh:
        st.markdown(
            """<a href="https://github.com/nsnatthapat/my-ml-project" target="_blank"
               style="display:inline-block; background:#1E3A5F; color:white;
               border-radius:8px; padding:12px 24px; font-weight:700;
               text-decoration:none; font-size:15px;">
              📂 View on GitHub →
            </a>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        | File | Purpose |
        |---|---|
        | `src/data/loader.py` | CSV ingestion, filename parsing, multi-site concatenation |
        | `src/data/quality.py` | Automated data quality gate with 5 checks |
        | `src/data/cleaner.py` | Dedup, forward-fill, dtype coercion → cleaned.csv |
        | `src/features/engineering.py` | 27 lag, rolling, temporal, and solar features |
        | `src/features/run_features.py` | End-to-end feature pipeline runner |
        | `src/models/baseline.py` | Persistence + LinearRegression baselines |
        | `src/models/compare_models.py` | Ridge / XGBoost / LightGBM CV comparison |
        | `src/models/compare_ts_models.py` | ARIMA / SARIMA classical TS models |
        | `src/models/tuning.py` | Optuna hyperparameter search |
        | `src/models/run_training.py` | MLflow-tracked final training pipeline |
        | `app/streamlit_app.py` | This portfolio dashboard |
        """
    )

    render_footer()
