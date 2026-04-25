import pandas as pd
import numpy as np
from pathlib import Path


def check_data_quality(
    df: pd.DataFrame,
    required_columns: list[str] = None,
    expected_dtypes: dict[str, str] = None,
    numeric_bounds: dict[str, tuple] = None,
    target_col: str = None,
) -> dict:
    """
    Run 5 data quality checks and return a structured report.

    Parameters
    ----------
    required_columns : columns that must be present
    expected_dtypes  : {col: dtype_str} e.g. {"Power(MW)": "float64"}
    numeric_bounds   : {col: (min, max)} — None means unbounded on that side
    target_col       : column to treat as classification target for check 5
    """
    failures = []
    warnings = []
    statistics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls_by_column": df.isnull().sum().to_dict(),
        "null_rate_by_column": (df.isnull().mean() * 100).round(2).to_dict(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
    }

    # ── Check 1: Schema validation ─────────────────────────────────────────────
    if required_columns:
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            failures.append(f"[Schema] Missing required columns: {missing_cols}")

    if expected_dtypes:
        for col, expected in expected_dtypes.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            if not actual.startswith(expected.rstrip("0123456789")):
                failures.append(
                    f"[Schema] Column '{col}' expected dtype '{expected}', got '{actual}'"
                )

    # ── Check 2: Row count ─────────────────────────────────────────────────────
    n = len(df)
    if n < 100:
        failures.append(f"[Row count] Only {n:,} rows — minimum 100 required.")
    elif n < 1000:
        warnings.append(f"[Row count] Only {n:,} rows — recommend at least 1,000.")

    # ── Check 3: Null rates ────────────────────────────────────────────────────
    null_rates = df.isnull().mean() * 100
    for col, rate in null_rates.items():
        if rate > 50:
            failures.append(
                f"[Nulls] '{col}' has {rate:.1f}% nulls (critical threshold: 50%)."
            )
        elif rate > 20:
            warnings.append(
                f"[Nulls] '{col}' has {rate:.1f}% nulls (warning threshold: 20%)."
            )

    # ── Check 4: Value ranges ──────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include="number")
    bounds = numeric_bounds or {}

    range_stats = {}
    for col in numeric_df.columns:
        col_min = float(numeric_df[col].min())
        col_max = float(numeric_df[col].max())
        col_std = float(numeric_df[col].std())
        range_stats[col] = {"min": col_min, "max": col_max, "std": col_std}

        lo, hi = bounds.get(col, (None, None))
        if lo is not None and col_min < lo:
            failures.append(
                f"[Range] '{col}' has values below {lo} (min observed: {col_min:.4f})."
            )
        if hi is not None and col_max > hi:
            failures.append(
                f"[Range] '{col}' has values above {hi} (max observed: {col_max:.4f})."
            )

        # Warn on near-zero variance (constant or near-constant column)
        if col_std == 0:
            warnings.append(f"[Range] '{col}' has zero variance (constant column).")
        elif col_std < 1e-6:
            warnings.append(f"[Range] '{col}' has near-zero variance (std={col_std:.2e}).")

    statistics["value_ranges"] = range_stats

    # ── Check 5: Target distribution ──────────────────────────────────────────
    if target_col:
        if target_col not in df.columns:
            failures.append(f"[Target] Column '{target_col}' not found in DataFrame.")
        else:
            value_counts = df[target_col].value_counts(normalize=True) * 100
            n_classes = len(value_counts)
            statistics["target_class_distribution"] = value_counts.round(2).to_dict()

            if n_classes < 2:
                failures.append(
                    f"[Target] '{target_col}' has only {n_classes} class — need at least 2."
                )
            else:
                rare = value_counts[value_counts < 5]
                if not rare.empty:
                    warnings.append(
                        f"[Target] '{target_col}' has {len(rare)} class(es) with < 5% of data: "
                        f"{rare.index.tolist()}"
                    )
                max_pct = float(value_counts.iloc[0])
                if max_pct > 80:
                    warnings.append(
                        f"[Target] '{target_col}' is imbalanced — dominant class is {max_pct:.1f}% of data."
                    )

    success = len(failures) == 0
    return {
        "success": success,
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }


def print_report(report: dict) -> None:
    status = "PASSED" if report["success"] else "FAILED"
    print(f"\n{'=' * 60}")
    print(f"DATA QUALITY GATE: {status}")
    print(f"{'=' * 60}")

    stats = report["statistics"]
    print(f"\nRows: {stats['total_rows']:,}  |  Columns: {stats['total_columns']}")

    if report["failures"]:
        print(f"\nCRITICAL FAILURES ({len(report['failures'])}):")
        for f in report["failures"]:
            print(f"  [FAIL] {f}")

    if report["warnings"]:
        print(f"\nWARNINGS ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  [WARN] {w}")

    if not report["failures"] and not report["warnings"]:
        print("\n  All checks passed with no warnings.")

    print("\nNull rates:")
    for col, rate in stats["null_rate_by_column"].items():
        print(f"  {col}: {rate:.1f}%")

    if "value_ranges" in stats:
        print("\nValue ranges:")
        for col, r in stats["value_ranges"].items():
            print(f"  {col}: min={r['min']:.4f}, max={r['max']:.4f}, std={r['std']:.4f}")

    if "target_class_distribution" in stats:
        print("\nTarget class distribution:")
        for cls, pct in stats["target_class_distribution"].items():
            print(f"  {cls}: {pct:.1f}%")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from data.loader import load_all

    import argparse

    parser = argparse.ArgumentParser(description="Run data quality gate on solar PV dataset")
    parser.add_argument("--prefix", choices=("Actual", "DA", "HA4"), default="Actual")
    parser.add_argument("--type", dest="pv_type", choices=("DPV", "UPV"), default=None)
    args = parser.parse_args()

    print(f"Loading data (prefix={args.prefix}, type={args.pv_type or 'all'})...")
    df = load_all(prefix=args.prefix, pv_type=args.pv_type)

    report = check_data_quality(
        df,
        required_columns=["LocalTime", "Power(MW)"],
        expected_dtypes={"Power(MW)": "float64", "LocalTime": "datetime64"},
        numeric_bounds={
            "Power(MW)": (0, None),   # power cannot be negative
            "lat": (24.0, 50.0),      # contiguous US latitude range
            "lon": (-125.0, -66.0),   # contiguous US longitude range
            "year": (2000, 2030),
        },
    )

    print_report(report)
    sys.exit(0 if report["success"] else 1)
