import numpy as np
import pandas as pd
import pytest

from features.engineering import create_features, select_features

# Expected final column count:
#   Original 9 cols − dropped 2 (year, capacity) + 27 engineered = 34
_EXPECTED_COLS = 34

# Lag features introduce NaN at the start of each site's series.
# These columns are intentionally excluded from the no-NaN check.
_LAG_COLS = {
    "power_lag_1", "power_lag_2", "power_lag_4", "power_lag_12", "power_lag_24",
    "rolling_std_12", "rolling_mean_12", "rolling_mean_288",
}


# ── Fixture ───────────────────────────────────────────────────────────────────

def _raw_df(n_per_site: int = 50, n_sites: int = 2) -> pd.DataFrame:
    """
    Minimal DataFrame matching the raw loader output, sorted by
    [lat, lon, LocalTime] as create_features() requires.
    """
    rng    = np.random.default_rng(42)
    sites  = [
        {"lat": 47.05, "lon": -122.25, "pv_type": "DPV", "capacity": "4MW"},
        {"lat": 46.75, "lon": -120.05, "pv_type": "UPV", "capacity": "109MW"},
    ][:n_sites]

    chunks = []
    for site in sites:
        times = pd.date_range("2006-06-01", periods=n_per_site, freq="5min")
        chunk = pd.DataFrame({
            "LocalTime" : times,
            "Power(MW)" : rng.uniform(0.0, float(site["capacity"].replace("MW", "")), n_per_site),
            "prefix"    : "Actual",
            "lat"       : site["lat"],
            "lon"       : site["lon"],
            "year"      : 2006,
            "pv_type"   : site["pv_type"],
            "capacity"  : site["capacity"],
            "interval"  : "5_Min",
        })
        chunks.append(chunk)

    return (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["lat", "lon", "LocalTime"])
        .reset_index(drop=True)
    )


# ── Column count ──────────────────────────────────────────────────────────────

class TestColumnCount:
    def test_expected_column_count(self):
        out = create_features(_raw_df())
        assert len(out.columns) == _EXPECTED_COLS, (
            f"Expected {_EXPECTED_COLS} columns, got {len(out.columns)}.\n"
            f"Columns: {sorted(out.columns.tolist())}"
        )

    def test_original_cols_dropped(self):
        out = create_features(_raw_df())
        assert "year"     not in out.columns
        assert "capacity" not in out.columns

    def test_expected_new_cols_present(self):
        out = create_features(_raw_df())
        for col in ["hour", "month", "season", "is_daytime",
                    "hour_sin", "hour_cos", "month_sin", "month_cos",
                    "capacity_mw", "daytime_capacity", "solar_exposure_index",
                    "power_lag_1", "rolling_std_12", "rolling_mean_288"]:
            assert col in out.columns, f"Expected column '{col}' not found"

    def test_custom_lag_periods_change_col_count(self):
        # lag_periods=[1, 2] → 2 lags instead of 5 → 3 fewer columns
        out = create_features(_raw_df(), lag_periods=[1, 2])
        assert len(out.columns) == _EXPECTED_COLS - 3

    def test_input_not_mutated(self):
        df  = _raw_df()
        original_cols = df.columns.tolist()
        create_features(df)
        assert df.columns.tolist() == original_cols


# ── NaN values ────────────────────────────────────────────────────────────────

class TestNoNaN:
    def test_non_lag_cols_have_no_nan(self):
        out  = create_features(_raw_df(n_per_site=60))
        check = [c for c in out.columns if c not in _LAG_COLS]
        null_counts = out[check].isnull().sum()
        bad = null_counts[null_counts > 0]
        assert bad.empty, f"Unexpected NaNs in non-lag columns:\n{bad}"

    def test_lag_cols_only_nan_at_start_of_each_site(self):
        # After the warm-up period (lag_24 = 24 rows), lags should be non-null.
        out  = create_features(_raw_df(n_per_site=100))
        warm = 24   # largest lag
        # Check each site block separately
        for (lat, lon), grp in out.groupby(["lat", "lon"]):
            tail = grp.iloc[warm:]
            assert tail["power_lag_24"].isnull().sum() == 0, (
                f"NaN in power_lag_24 after warm-up at site lat={lat} lon={lon}"
            )


# ── Feature value ranges ──────────────────────────────────────────────────────

class TestFeatureRanges:
    @pytest.fixture(autouse=True)
    def _df(self):
        self.out = create_features(_raw_df(n_per_site=60))

    def test_hour_range(self):
        assert self.out["hour"].between(0, 23).all()

    def test_month_range(self):
        assert self.out["month"].between(1, 12).all()

    def test_season_range(self):
        assert self.out["season"].isin([0, 1, 2, 3]).all()

    def test_is_daytime_binary(self):
        assert self.out["is_daytime"].isin([0, 1]).all()

    def test_is_weekend_binary(self):
        assert self.out["is_weekend"].isin([0, 1]).all()

    def test_is_utility_scale_binary(self):
        assert self.out["is_utility_scale"].isin([0, 1]).all()

    def test_hour_sin_range(self):
        assert self.out["hour_sin"].between(-1.0, 1.0).all()

    def test_hour_cos_range(self):
        assert self.out["hour_cos"].between(-1.0, 1.0).all()

    def test_capacity_mw_positive(self):
        assert (self.out["capacity_mw"] > 0).all()

    def test_daytime_capacity_non_negative(self):
        assert (self.out["daytime_capacity"] >= 0).all()

    def test_daytime_capacity_zero_at_night(self):
        night = self.out[self.out["hour"].lt(6) | self.out["hour"].gt(20)]
        assert (night["daytime_capacity"] == 0).all()

    def test_capacity_normalized_power_clipped(self):
        valid = self.out["capacity_normalized_power"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_lat_normalized_range(self):
        assert self.out["lat_normalized"].between(0.0, 1.0).all()

    def test_capacity_tier_values(self):
        assert self.out["capacity_tier"].isin([0, 1, 2]).all()


# ── select_features ───────────────────────────────────────────────────────────

class TestSelectFeatures:
    def test_returns_tuple(self):
        df  = create_features(_raw_df(n_per_site=60))
        result = select_features(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_target_always_kept(self):
        df = create_features(_raw_df(n_per_site=60))
        names, reduced = select_features(df, target_col="Power(MW)")
        assert "Power(MW)" in names
        assert "Power(MW)" in reduced.columns

    def test_fewer_or_equal_cols_after_selection(self):
        df = create_features(_raw_df(n_per_site=60))
        names, reduced = select_features(df)
        assert len(reduced.columns) <= len(df.columns)
