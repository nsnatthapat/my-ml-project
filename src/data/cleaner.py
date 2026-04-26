import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLEANED_PATH = DATA_DIR / "cleaned.csv"


def clean_data(
    df: pd.DataFrame,
    target_col: str = None,
    time_col: str = None,
    is_classification: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean a DataFrame and run a quality gate on the result.

    Parameters
    ----------
    target_col        : column treated as the ML target (rows with null target are dropped)
    time_col          : datetime column — if provided, enables time-series mode (forward-fill
                        instead of row-drop for non-target nulls, sorts by time first)
    is_classification : pass True to run the class-distribution quality check on target_col

    Returns
    -------
    (cleaned_df, quality_report)
    """
    from data.quality import check_data_quality

    df = df.copy()
    is_timeseries = time_col is not None and time_col in df.columns

    # ── Step 1: Drop columns with > 50% nulls ─────────────────────────────────
    null_rates = df.isnull().mean()
    drop_cols = null_rates[null_rates > 0.5].index.tolist()
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} column(s) with >50% nulls: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # ── Step 2: Drop rows where target is null ─────────────────────────────────
    if target_col and target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped:,} row(s) with null target ('{target_col}').")

    # ── Step 3: Handle remaining nulls ────────────────────────────────────────
    if is_timeseries:
        df = df.sort_values(time_col).reset_index(drop=True)
        null_before = df.isnull().sum().sum()
        df = df.ffill()
        # Any remaining nulls at the start of series that ffill can't reach
        df = df.bfill()
        null_after = df.isnull().sum().sum()
        filled = null_before - null_after
        if filled:
            print(f"  Forward/back-filled {filled:,} null value(s) (time-series mode).")
    else:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped:,} row(s) containing nulls.")

    # ── Step 4: Remove exact duplicates ───────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Removed {dropped:,} duplicate row(s).")

    # ── Step 5: Convert dtypes ─────────────────────────────────────────────────
    for col in df.columns:
        if col == time_col:
            continue
        if df[col].dtype == object:
            # Try numeric coercion first; fall back to string
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() > 0.9:
                df[col] = coerced
                print(f"  Converted '{col}' object -> numeric.")
            else:
                df[col] = df[col].astype(str)
        elif df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(int)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(float)

    # ── Step 6: Save cleaned CSV ───────────────────────────────────────────────
    df.to_csv(CLEANED_PATH, index=False)
    print(f"  Saved cleaned data to {CLEANED_PATH}  ({len(df):,} rows)")

    # ── Step 7: Re-run quality gate ────────────────────────────────────────────
    expected_dtypes = {}
    for col in df.select_dtypes(include="number").columns:
        expected_dtypes[col] = "int" if pd.api.types.is_integer_dtype(df[col]) else "float"
    if time_col and time_col in df.columns:
        expected_dtypes[time_col] = "datetime64"

    quality_result = check_data_quality(
        df,
        required_columns=[c for c in [time_col, target_col] if c],
        expected_dtypes=expected_dtypes,
        target_col=target_col if is_classification else None,
    )

    return df, quality_result


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from data.loader import load_all
    from data.quality import print_report

    parser = argparse.ArgumentParser(description="Clean solar PV dataset")
    parser.add_argument("--prefix", choices=("Actual", "DA", "HA4"), default="Actual")
    parser.add_argument("--type", dest="pv_type", choices=("DPV", "UPV"), default=None)
    args = parser.parse_args()

    print(f"Loading raw data (prefix={args.prefix}, type={args.pv_type or 'all'})...")
    raw = load_all(prefix=args.prefix, pv_type=args.pv_type)
    print(f"  Raw shape:  {len(raw):,} rows x {len(raw.columns)} columns\n")

    print("Cleaning...")
    cleaned, quality = clean_data(raw, target_col="Power(MW)", time_col="LocalTime")

    print(f"\nBefore: {len(raw):,} rows")
    print(f"After:  {len(cleaned):,} rows  ({len(raw) - len(cleaned):,} removed)\n")

    print_report(quality)
    sys.exit(0 if quality["success"] else 1)
