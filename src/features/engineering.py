"""
Feature engineering for multi-step ahead solar PV generation forecasting.
Target: Power(MW) — continuous regression, per-site, N steps ahead.

Run directly to produce data/features.csv:
    python src/features/engineering.py [--force-reload]
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_DEFAULT_LAGS = [1, 2, 4, 12, 24]

_SEASON_MAP = {
    12: 0, 1: 0, 2: 0,   # winter
     3: 1, 4: 1, 5: 1,   # spring
     6: 2, 7: 2, 8: 2,   # summer
     9: 3, 10: 3, 11: 3  # fall
}


def create_features(
    df: pd.DataFrame,
    lag_periods: list[int] = None,
) -> pd.DataFrame:
    """
    Engineer 23 predictive features from the cleaned solar PV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_all() + clean_data(). Must be sorted by
        ["lat", "lon", "LocalTime"] before calling this function so
        that per-site lag/rolling features are computed correctly.
        Required columns: LocalTime, Power(MW), lat, lon, pv_type,
        capacity, prefix, interval, year.

    lag_periods : list[int], optional
        Lag steps to compute for Power(MW). Defaults to [1, 2, 4, 12, 24].
        For 5-min Actual data: lag_12 = 1 hour ago, lag_24 = 2 hours ago.

    Returns
    -------
    pd.DataFrame
        New DataFrame (input is NOT mutated) with engineered features
        appended. Drops 'year' (constant) and 'capacity' (replaced by
        numeric capacity_mw).
    """
    if lag_periods is None:
        lag_periods = _DEFAULT_LAGS

    out = df.copy()

    # ── Prerequisite: parse capacity string → float ────────────────────────────
    # Must be computed first because three interaction features and the
    # capacity_tier ordinal all depend on it.
    out["capacity_mw"] = out["capacity"].str.replace("MW", "").astype(float)

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 1: Temporal / Domain Features
    #
    # Solar irradiance follows a strict diurnal and seasonal cycle. Raw
    # integer encodings of hour and month break the circular structure
    # (hour 23 is adjacent to hour 0; December is adjacent to January).
    # Sin/cos projections preserve topology and are numerically stable for
    # gradient-based models. For multi-step-ahead forecasting these features
    # serve as the "clock" the model uses to anticipate upcoming production.
    # ══════════════════════════════════════════════════════════════════════════

    # Hour of day (0–23). Strongest single temporal predictor: power is
    # near zero outside ~06:00–20:00 and peaks sharply at 11–14.
    out["hour"] = out["LocalTime"].dt.hour

    # Calendar month (1–12). Captures seasonal variation in day length;
    # Washington State (45–49°N) sees ~5-hour swings between winter and summer.
    out["month"] = out["LocalTime"].dt.month

    # Day of year (1–366). Finer-grained seasonal signal than month; tree
    # models can split at specific day-of-year thresholds around solstices.
    out["day_of_year"] = out["LocalTime"].dt.day_of_year

    # Weekend binary flag. Grid load and dispatch scheduling differ on
    # weekends; relevant for DA/HA4 forecast horizons used in unit commitment.
    out["is_weekend"] = out["LocalTime"].dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical hour encoding — sine/cosine pair on a 24-hour circle.
    # Without this, a linear model perceives hour 23 as 23 steps from hour 0
    # rather than 1 step, introducing a discontinuity at midnight.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    # Cyclical month encoding — sine/cosine pair on a 12-month circle.
    # Ensures December and January are adjacent in the feature space.
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)

    # Daytime binary flag. Hours 6–20 encompass essentially all non-zero
    # production at Washington State latitudes. Acts as a hard gate for the
    # ~70% zero-inflation: a two-stage model can use this as its first-stage
    # binary target (is_daytime → run regression, else predict 0).
    out["is_daytime"] = out["hour"].between(6, 20).astype(int)

    # Season ordinal (0=winter, 1=spring, 2=summer, 3=fall). Discrete
    # seasonal bucket for models that benefit from low-cardinality
    # categoricals (e.g. LightGBM native categoricals). Meteorological
    # seasons: Dec–Feb=winter, Mar–May=spring, Jun–Aug=summer, Sep–Nov=fall.
    out["season"] = out["month"].map(_SEASON_MAP)

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 2: Site / Capacity Features
    #
    # EDA confirmed that site nameplate capacity is the strongest predictor
    # of absolute output. DPV (distributed, 0.2–5 MW) and UPV (utility,
    # 1–109 MW) sites operate at completely different scales; encoding their
    # identity and size explicitly prevents the model from treating a 0.3 MW
    # rooftop the same as a 109 MW solar farm.
    # ══════════════════════════════════════════════════════════════════════════

    # Binary utility-scale flag. UPV sites have better panel orientation,
    # less shading, and disproportionately higher output than DPV sites.
    # A single bit captures this structural split cleanly.
    out["is_utility_scale"] = (out["pv_type"] == "UPV").astype(int)

    # Capacity utilisation ratio (0–1). Normalises Power(MW) by nameplate
    # rating so sites of different sizes are comparable on the same scale.
    # ⚠️  TARGET-DERIVED FEATURE: drop from X before model training/inference
    #     (the target Power(MW) is unknown at prediction time).
    #     Retain here for analysis, post-hoc inspection, and two-stage validation.
    out["capacity_normalized_power"] = (
        out["Power(MW)"] / out["capacity_mw"]
    ).clip(0, 1)

    # Normalised latitude — scales observed lat range to [0, 1].
    # Uses the dataset's own min/max so the feature is dimensionless.
    _lat_min = out["lat"].min()
    _lat_max = out["lat"].max()
    if _lat_max > _lat_min:
        out["lat_normalized"] = (out["lat"] - _lat_min) / (_lat_max - _lat_min)
    else:
        out["lat_normalized"] = 0.5   # degenerate: all sites at same latitude

    # Capacity tier ordinal. Three-level buckets: small (<1 MW), medium
    # (1–10 MW), large (>10 MW). Tree models split on this ordinal efficiently
    # without needing one-hot encoding.
    out["capacity_tier"] = pd.cut(
        out["capacity_mw"],
        bins=[0, 1, 10, float("inf")],
        labels=[0, 1, 2],   # 0=small, 1=medium, 2=large
        right=True,
    ).astype(int)

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 3: Interaction / Composite Features
    #
    # Solar output is jointly determined by time-of-day AND site scale. A
    # 109 MW farm and a 0.3 MW rooftop respond differently to the same solar
    # angle. Encoding products of these dimensions lets linear models capture
    # non-linear relationships that individual features cannot express alone.
    # ══════════════════════════════════════════════════════════════════════════

    # Daytime × capacity. During night, this is zero regardless of capacity —
    # directly encoding the physical fact that nameplate rating only matters
    # when the sun is up. For tree models this pre-computes a key split.
    out["daytime_capacity"] = out["is_daytime"] * out["capacity_mw"]

    # Peak hour × utility scale. Hours 11–14 are the peak window AND UPV
    # sites show the steepest ramp rate. The product captures the joint
    # extreme: a large site at solar noon produces disproportionately more
    # than either factor alone would suggest.
    out["peak_hour_upv"] = (
        out["hour"].between(11, 14).astype(int) * out["is_utility_scale"]
    )

    # Latitude × hour_cos. hour_cos peaks at 1.0 at solar noon and falls
    # symmetrically. Multiplied by raw lat, this captures the physical
    # attenuation: for the same hour, higher latitudes receive lower solar
    # irradiance due to a greater zenith angle.
    out["lat_hour_interaction"] = out["lat"] * out["hour_cos"]

    # Solar exposure index — composite score: capacity × solar angle ×
    # latitude penalty centred at 47°N (the study-region centroid).
    # The lat term penalises sites >3° from 47°N, clipped at 0 to avoid
    # negative values. Effective range: [0, capacity_mw].
    out["solar_exposure_index"] = (
        out["capacity_mw"]
        * out["hour_cos"]
        * (1 - (out["lat"] - 47).abs() / 3).clip(lower=0)
    )

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 4: Lag / Rolling Features (per-site)
    #
    # Multi-step-ahead forecasting requires the model to have recent output
    # history as context. Without lag features, the model has no "memory"
    # of what the site was generating moments ago and cannot detect ramps,
    # cloud events, or morning ramp-up. All lags and rolling windows are
    # computed per-site (groupby lat/lon) to ensure lag_1 of site A never
    # picks up the last reading of site B.
    #
    # The calling code MUST sort df by ["lat", "lon", "LocalTime"] before
    # calling create_features() so shifts are applied in time order.
    # ══════════════════════════════════════════════════════════════════════════

    site_groups = out.groupby(["lat", "lon"], sort=False)["Power(MW)"]

    for n in lag_periods:
        # Lag N: power reading N timesteps ago at this site.
        # For 5-min Actual data: lag_12 = 1 hour ago, lag_24 = 2 hours ago.
        # For 60-min DA/HA4 data: lag_12 = 12 hours ago, lag_24 = 1 day ago.
        out[f"power_lag_{n}"] = site_groups.shift(n)

    # Rolling mean over the last 12 timesteps (1 hour for 5-min data).
    # shift(1) inside the lambda ensures the window never includes the current
    # row, preventing data leakage from target into the feature.
    # Captures the short-term trend: is output rising or falling?
    out["rolling_mean_12"] = site_groups.transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean()
    )

    # Rolling standard deviation over the same 12-step window.
    # High std signals cloud cover, passing weather, or a morning ramp event —
    # all of which have predictive value for what happens in the next interval.
    out["rolling_std_12"] = site_groups.transform(
        lambda s: s.shift(1).rolling(12, min_periods=2).std()
    )

    # 24-hour rolling mean (288 timesteps for 5-min data, 24 for 60-min).
    # Approximates "same time yesterday" and captures daily seasonality —
    # a site that was generating 5 MW at this hour yesterday will likely
    # generate a similar amount today under similar conditions.
    out["rolling_mean_288"] = site_groups.transform(
        lambda s: s.shift(1).rolling(288, min_periods=1).mean()
    )

    # ── Drop constant / replaced columns ──────────────────────────────────────
    # 'year' is constant across the entire dataset (2006 only).
    # 'capacity' is a raw string; replaced by the numeric capacity_mw.
    out = out.drop(columns=["year", "capacity"])

    return out


def select_features(
    df: pd.DataFrame,
    corr_threshold: float = 0.95,
    variance_threshold_pct: float = 0.01,
    target_col: str = "Power(MW)",
) -> tuple[list[str], pd.DataFrame]:
    """
    Select informative features by removing highly correlated and low-variance
    numeric columns. Non-numeric columns are always kept unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Output of create_features(). Must contain numeric columns to evaluate.
    corr_threshold : float
        Absolute Pearson correlation above which the second feature in a pair
        is considered redundant and dropped. Default 0.95.
    variance_threshold_pct : float
        Fraction of the mean feature variance below which a feature is
        considered near-constant and dropped. Default 0.01 (1%).
        variance_threshold = variance_threshold_pct × mean(all feature variances).
    target_col : str
        Target column to protect — never dropped, excluded from checks.

    Returns
    -------
    (selected_feature_names, reduced_df)
        selected_feature_names : list[str] — numeric columns that survived both
            filters, plus target_col (if present).
        reduced_df : pd.DataFrame — df with redundant/low-variance columns removed.
    """
    # Identify numeric candidates — exclude the target to protect it from
    # being dropped, and exclude non-numeric columns (they are kept as-is).
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    candidates = [c for c in numeric_cols if c != target_col]

    # Work on a copy; track dropped columns with reasons for logging.
    drop_log: list[tuple[str, str]] = []   # (column_name, reason)

    # ── Step 1: Correlation filter ─────────────────────────────────────────────
    # Compute absolute Pearson correlation matrix on candidate columns only,
    # dropping rows with NaN so correlations are not inflated by missing lags.
    corr_df = df[candidates].dropna().corr().abs()

    # Inspect only the upper triangle (i < j) so each pair is seen once.
    # When |corr(A, B)| > threshold we drop B (the later column), keeping A.
    upper = corr_df.where(
        np.triu(np.ones(corr_df.shape, dtype=bool), k=1)
    )

    corr_drop: set[str] = set()
    for col in upper.columns:
        # Find all earlier columns that are highly correlated with this one.
        correlated_with = upper.index[upper[col] > corr_threshold].tolist()
        if correlated_with:
            partner = correlated_with[0]   # most prominent correlate
            r = corr_df.loc[partner, col]
            drop_log.append((
                col,
                f"correlation {r:.3f} > {corr_threshold} with '{partner}'"
            ))
            corr_drop.add(col)

    survivors_after_corr = [c for c in candidates if c not in corr_drop]

    # ── Step 2: Variance filter ────────────────────────────────────────────────
    # Compute per-feature variance on the correlation-filtered set.
    # overall_variance = mean variance across all surviving candidates, giving
    # the threshold a scale that adapts to the feature magnitudes in the data.
    variances = df[survivors_after_corr].var()
    overall_variance = variances.mean()
    threshold = variance_threshold_pct * overall_variance

    var_drop: set[str] = set()
    for col, var in variances.items():
        if var < threshold:
            drop_log.append((
                col,
                f"variance {var:.6f} < threshold {threshold:.6f} "
                f"({variance_threshold_pct*100:.1f}% of mean variance {overall_variance:.6f})"
            ))
            var_drop.add(col)

    survivors = [c for c in survivors_after_corr if c not in var_drop]

    # ── Log results ───────────────────────────────────────────────────────────
    total_dropped = len(corr_drop) + len(var_drop)
    print("\nFeature selection summary:")
    print(f"  Input features  : {len(candidates)}")
    print(f"  Correlation drops: {len(corr_drop)}")
    print(f"  Variance drops  : {len(var_drop)}")
    print(f"  Output features : {len(survivors)}")

    if drop_log:
        print(f"\n  Dropped ({total_dropped}):")
        for col, reason in drop_log:
            print(f"    ✗ {col!r:35s}  — {reason}")
    else:
        print("\n  No features dropped.")

    # ── Build output ──────────────────────────────────────────────────────────
    # selected_feature_names = surviving numeric features + target (if present)
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    selected_feature_names = survivors + (
        [target_col] if target_col in df.columns else []
    )

    # Keep surviving numeric features + target + all non-numeric columns
    keep_cols = [c for c in df.columns if c in set(survivors) or c == target_col
                 or c in set(non_numeric)]
    reduced_df = df[keep_cols]

    return selected_feature_names, reduced_df


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from data.loader import load_all
    from data.cleaner import clean_data

    CLEANED_PATH = DATA_DIR / "cleaned.csv"
    FEATURES_PATH = DATA_DIR / "features.csv"

    parser = argparse.ArgumentParser(
        description="Engineer features for solar PV generation forecasting"
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Ignore cleaned.csv and reload from raw CSVs via load_all() + clean_data()",
    )
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    if not args.force_reload and CLEANED_PATH.exists():
        print(f"Loading cleaned data from {CLEANED_PATH} ...")
        df_raw = pd.read_csv(CLEANED_PATH, parse_dates=["LocalTime"])
        print(f"  Loaded {len(df_raw):,} rows x {len(df_raw.columns)} columns")
    else:
        print("Reloading raw data via load_all() ...")
        df_loaded = load_all(prefix="Actual")
        print(f"  Raw shape: {df_loaded.shape}")
        print("Cleaning ...")
        df_raw, quality = clean_data(
            df_loaded, target_col="Power(MW)", time_col="LocalTime"
        )
        print(f"  Cleaned shape: {df_raw.shape}")

    # Sort by site then time — required for correct per-site lag computation.
    print("\nSorting by [lat, lon, LocalTime] ...")
    df_raw = df_raw.sort_values(["lat", "lon", "LocalTime"]).reset_index(drop=True)

    # ── Engineer features ──────────────────────────────────────────────────────
    print("Engineering features ...")
    cols_before = set(df_raw.columns)
    df_features = create_features(df_raw)
    cols_after = set(df_features.columns)

    new_cols = sorted(cols_after - cols_before)
    dropped_cols = sorted(cols_before - cols_after)

    print(f"\n  Columns before : {len(cols_before)}")
    print(f"  Columns after  : {len(cols_after)}")
    print(f"  Added   ({len(new_cols):2d}) : {new_cols}")
    print(f"  Dropped ({len(dropped_cols):2d}) : {dropped_cols}")

    # Show null counts for lag columns — expected NaNs at the start of each
    # site's series (the first N rows per site have no history yet).
    lag_cols = [c for c in df_features.columns if c.startswith("power_lag_")]
    print("\n  Null counts for lag / rolling columns (expected at series starts):")
    for col in lag_cols + ["rolling_mean_12", "rolling_std_12", "rolling_mean_288"]:
        print(f"    {col}: {df_features[col].isna().sum():,}")

    # ── Select features ───────────────────────────────────────────────────────
    print("\nSelecting features ...")
    selected_names, df_selected = select_features(df_features)
    print(f"\n  Final selected feature names ({len(selected_names)}):")
    for name in sorted(selected_names):
        print(f"    {name}")

    # ── Save ───────────────────────────────────────────────────────────────────
    df_selected.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved to {FEATURES_PATH}  ({len(df_selected):,} rows x {len(df_selected.columns)} cols)")
