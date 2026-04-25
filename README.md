# my-ml-project

Production ML project for solar PV generation forecasting (Washington State, 2006).

## Project Structure

```
src/data/       # loading, quality gate, cleaning
src/features/   # feature engineering
src/models/     # training and prediction
app/            # FastAPI + Streamlit
tests/          # unit tests
notebooks/      # EDA
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Exploratory Data Analysis

**Dataset:** 228 CSV files — 76 each of `Actual` (5-min), `DA` (day-ahead 60-min), and `HA4` (hour-ahead 60-min) forecasts. Covering 76 sites across Washington State. ~8M rows per prefix. Features: `LocalTime`, `Power(MW)`, `lat`, `lon`, `pv_type` (DPV/UPV), `capacity_mw`, `hour`, `month`.

**Key findings:**

- **~70% of readings are zero** — solar output is zero at night, producing a heavily bimodal distribution; models must handle this zero-inflation (e.g., two-stage model or zero-inflated regression).
- **Site capacity is the strongest predictor** — UPV (utility-scale) sites produce orders-of-magnitude more power than DPV (distributed); capacity and pv_type should be primary features.
- **Hour of day dominates temporally** — output peaks at hours 11–14 and is zero outside daylight; hour should be encoded cyclically (sin/cos), not as a raw integer.
- **Latitude has a weak negative effect** — northerly sites generate slightly less power; worth including but low signal relative to capacity and time features.
- **No missing data** — zero imputation required; `year` is constant (2006 only) and should be dropped before training.
