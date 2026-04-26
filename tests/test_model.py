import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "production_model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="production_model.pkl not found — run src/models/run_training.py first",
)

FEATURES = [
    "lon", "capacity_mw", "hour", "month", "season",
    "daytime_capacity", "solar_exposure_index",
    "power_lag_1", "power_lag_12", "power_lag_24",
    "rolling_std_12", "rolling_mean_288",
]

MAX_CAPACITY_MW = 109.0   # largest site in the dataset


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def artifact():
    return joblib.load(MODEL_PATH)


def _input_row(**overrides) -> pd.DataFrame:
    """Single-row DataFrame with sensible daytime defaults."""
    defaults = {
        "lon"                  : -120.05,
        "capacity_mw"          : 5.0,
        "hour"                 : 13,
        "month"                : 7,
        "season"               : 2,
        "daytime_capacity"     : 5.0,
        "solar_exposure_index" : 3.5,
        "power_lag_1"          : 3.2,
        "power_lag_12"         : 2.8,
        "power_lag_24"         : 2.5,
        "rolling_std_12"       : 0.4,
        "rolling_mean_288"     : 2.8,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])[FEATURES]


# ── Load & structure ──────────────────────────────────────────────────────────

class TestModelLoads:
    def test_artifact_has_required_keys(self, artifact):
        assert "model"    in artifact
        assert "features" in artifact

    def test_feature_list_matches_expected(self, artifact):
        assert artifact["features"] == FEATURES

    def test_model_has_predict_method(self, artifact):
        assert callable(getattr(artifact["model"], "predict", None))


# ── Prediction correctness ────────────────────────────────────────────────────

class TestPredictions:
    def test_single_row_prediction_shape(self, artifact):
        pred = artifact["model"].predict(_input_row())
        assert pred.shape == (1,)

    def test_batch_prediction_shape(self, artifact):
        rows = pd.concat([_input_row()] * 10, ignore_index=True)
        pred = artifact["model"].predict(rows)
        assert pred.shape == (10,)

    def test_predictions_non_negative(self, artifact):
        pred = artifact["model"].predict(_input_row())
        assert float(pred[0]) >= 0.0, f"Got negative prediction: {pred[0]}"

    def test_predictions_within_physical_max(self, artifact):
        pred = artifact["model"].predict(_input_row())
        assert float(pred[0]) <= MAX_CAPACITY_MW, (
            f"Prediction {pred[0]:.3f} MW exceeds max site capacity {MAX_CAPACITY_MW} MW"
        )

    def test_nighttime_prediction_near_zero(self, artifact):
        night = _input_row(
            hour=2, daytime_capacity=0.0, solar_exposure_index=0.0,
            power_lag_1=0.0, power_lag_12=0.0, power_lag_24=0.0,
            rolling_std_12=0.0, rolling_mean_288=0.0,
        )
        pred = float(artifact["model"].predict(night)[0])
        assert pred < 1.0, f"Night prediction should be near zero, got {pred:.4f} MW"

    def test_daytime_prediction_positive(self, artifact):
        day = _input_row(
            hour=13, month=7, daytime_capacity=50.0,
            solar_exposure_index=35.0, capacity_mw=50.0,
            power_lag_1=25.0, power_lag_12=22.0, power_lag_24=20.0,
            rolling_mean_288=21.0,
        )
        pred = float(artifact["model"].predict(day)[0])
        assert pred > 0.0, f"Peak-hour daytime prediction should be > 0, got {pred:.4f} MW"

    def test_larger_capacity_gives_larger_prediction(self, artifact):
        small = artifact["model"].predict(_input_row(capacity_mw=1.0,  daytime_capacity=1.0))[0]
        large = artifact["model"].predict(_input_row(capacity_mw=50.0, daytime_capacity=50.0))[0]
        assert large > small, (
            f"Larger-capacity site should predict more power "
            f"(small={small:.3f}, large={large:.3f})"
        )

    def test_prediction_deterministic(self, artifact):
        pred_a = artifact["model"].predict(_input_row())[0]
        pred_b = artifact["model"].predict(_input_row())[0]
        assert pred_a == pred_b
