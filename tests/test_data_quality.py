import numpy as np
import pandas as pd
import pytest

from data.quality import check_data_quality


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _clean_df(n: int = 200) -> pd.DataFrame:
    """Minimal well-formed solar PV DataFrame that should pass all checks."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "LocalTime" : pd.date_range("2006-01-01", periods=n, freq="5min"),
        "Power(MW)" : rng.uniform(0.0, 5.0, n),
        "lat"       : rng.uniform(45.0, 49.0, n),
        "lon"       : rng.uniform(-123.0, -117.0, n),
        "capacity_mw": rng.uniform(0.1, 10.0, n),
    })


REQUIRED = ["LocalTime", "Power(MW)", "lat", "lon", "capacity_mw"]
DTYPES   = {"Power(MW)": "float64", "lat": "float64", "capacity_mw": "float64"}
BOUNDS   = {"Power(MW)": (0.0, None), "capacity_mw": (0.0, None)}


# ── Pass: clean data ──────────────────────────────────────────────────────────

class TestQualityGatePasses:
    def test_success_flag(self):
        result = check_data_quality(_clean_df(), REQUIRED, DTYPES, BOUNDS)
        assert result["success"] is True

    def test_no_failures(self):
        result = check_data_quality(_clean_df(), REQUIRED, DTYPES, BOUNDS)
        assert result["failures"] == []

    def test_statistics_keys_present(self):
        result = check_data_quality(_clean_df(), REQUIRED, DTYPES, BOUNDS)
        stats = result["statistics"]
        assert "total_rows" in stats
        assert "total_columns" in stats
        assert "total_nulls_by_column" in stats

    def test_row_count_recorded(self):
        df = _clean_df(n=300)
        result = check_data_quality(df, REQUIRED, DTYPES, BOUNDS)
        assert result["statistics"]["total_rows"] == 300

    def test_return_keys(self):
        result = check_data_quality(_clean_df())
        assert set(result.keys()) == {"success", "failures", "warnings", "statistics"}


# ── Fail: broken datasets ─────────────────────────────────────────────────────

class TestQualityGateCatchesBadData:
    def test_catches_missing_required_column(self):
        df = _clean_df().drop(columns=["Power(MW)"])
        result = check_data_quality(df, required_columns=REQUIRED)
        assert result["success"] is False
        assert any("Power(MW)" in msg for msg in result["failures"])

    def test_catches_multiple_missing_columns(self):
        df = _clean_df().drop(columns=["lat", "lon"])
        result = check_data_quality(df, required_columns=REQUIRED)
        assert result["success"] is False
        failures_text = " ".join(result["failures"])
        assert "lat" in failures_text or "lon" in failures_text

    def test_catches_too_few_rows(self):
        df = _clean_df().head(50)   # below the 100-row minimum
        result = check_data_quality(df, REQUIRED, DTYPES, BOUNDS)
        assert result["success"] is False
        assert any("rows" in msg.lower() for msg in result["failures"])

    def test_catches_excessive_nulls(self):
        df = _clean_df(n=200)
        df.loc[:, "Power(MW)"] = np.where(
            np.arange(200) < 120, np.nan, df["Power(MW)"]
        )  # 60% nulls — above 50% threshold
        result = check_data_quality(df, REQUIRED, DTYPES, BOUNDS)
        assert result["success"] is False
        assert any("Power(MW)" in msg for msg in result["failures"])

    def test_catches_wrong_dtype(self):
        df = _clean_df(n=200)
        df["Power(MW)"] = df["Power(MW)"].astype(str)  # string instead of float
        result = check_data_quality(df, REQUIRED, {"Power(MW)": "float64"}, BOUNDS)
        assert result["success"] is False
        assert any("Power(MW)" in msg for msg in result["failures"])

    def test_catches_out_of_range_values(self):
        df = _clean_df(n=200)
        df.loc[0, "Power(MW)"] = -999.0   # negative power is physically impossible
        result = check_data_quality(
            df, REQUIRED, DTYPES,
            numeric_bounds={"Power(MW)": (0.0, None)},
        )
        assert result["success"] is False
        assert any("Power(MW)" in msg for msg in result["failures"])

    def test_warns_on_low_row_count(self):
        df = _clean_df().head(500)   # above 100 but below 1000 → warning, not failure
        result = check_data_quality(df, REQUIRED, DTYPES, BOUNDS)
        assert result["success"] is True
        assert any("rows" in w.lower() for w in result["warnings"])
