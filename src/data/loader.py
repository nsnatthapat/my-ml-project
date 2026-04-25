import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PREFIXES = ("Actual", "DA", "HA4")
TYPES = ("DPV", "UPV")


def parse_filename(path: Path) -> dict:
    """Extract metadata encoded in the filename."""
    stem = path.stem
    parts = stem.split("_")
    # e.g. Actual_45.65_-122.55_2006_DPV_0.3MW_5_Min
    try:
        return {
            "prefix": parts[0],
            "lat": float(parts[1]),
            "lon": float(parts[2]),
            "year": int(parts[3]),
            "pv_type": parts[4],
            "capacity": parts[5],
            "interval": f"{parts[6]}_{parts[7]}",
        }
    except (IndexError, ValueError):
        return {"prefix": parts[0] if parts else "unknown"}


def list_files(prefix: str = None, pv_type: str = None) -> list[Path]:
    """Return sorted list of CSVs, optionally filtered by prefix and/or pv_type."""
    files = sorted(DATA_DIR.glob("*.csv"))
    if prefix:
        files = [f for f in files if f.name.startswith(prefix)]
    if pv_type:
        files = [f for f in files if f"_{pv_type}_" in f.name]
    return files


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["LocalTime"] = pd.to_datetime(df["LocalTime"], format="%m/%d/%y %H:%M")
    meta = parse_filename(path)
    for k, v in meta.items():
        df[k] = v
    return df


def load_all(prefix: str = None, pv_type: str = None) -> pd.DataFrame:
    """Load and concatenate all matching CSVs into one DataFrame."""
    files = list_files(prefix=prefix, pv_type=pv_type)
    if not files:
        raise FileNotFoundError(f"No CSVs found (prefix={prefix}, pv_type={pv_type})")
    return pd.concat([load_csv(f) for f in files], ignore_index=True)


# ── profile helpers ────────────────────────────────────────────────────────────

def print_inventory() -> None:
    files = list_files()
    total = len(files)
    print(f"Total CSV files: {total}")
    print("\nBy prefix:")
    for p in PREFIXES:
        n = len(list_files(prefix=p))
        print(f"  {p:8s}: {n} files")
    print("\nBy PV type:")
    for t in TYPES:
        n = len(list_files(pv_type=t))
        print(f"  {t:5s}: {n} files")
    print("\nBy interval:")
    for interval in ("5_Min", "60_Min"):
        n = sum(1 for f in files if interval in f.name)
        print(f"  {interval}: {n} files")


def print_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"\nShape: {rows:,} rows x {cols} columns")


def print_column_info(df: pd.DataFrame) -> None:
    print("\nColumns and data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")


def print_summary_statistics(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("\nNo numeric columns found.")
        return
    stats = numeric_df.agg(["mean", "std", "min", "max"])
    print("\nSummary statistics (numeric columns):")
    print(stats.to_string())


def print_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing value counts:")
    if missing.empty:
        print("  No missing values.")
        return
    total = len(df)
    for col, count in missing.items():
        print(f"  {col}: {count:,} ({count / total * 100:.1f}%)")


def profile(df: pd.DataFrame) -> None:
    print_shape(df)
    print_column_info(df)
    print_summary_statistics(df)
    print_missing_values(df)


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect solar PV dataset")
    parser.add_argument("--prefix", choices=PREFIXES, help="Filter by prefix (Actual/DA/HA4)")
    parser.add_argument("--type", dest="pv_type", choices=TYPES, help="Filter by PV type (DPV/UPV)")
    parser.add_argument("--inventory", action="store_true", help="Show file inventory only")
    args = parser.parse_args()

    print("=" * 60)
    print("FILE INVENTORY")
    print("=" * 60)
    print_inventory()

    if args.inventory:
        raise SystemExit(0)

    print("\n" + "=" * 60)
    label = f"prefix={args.prefix or 'all'}, type={args.pv_type or 'all'}"
    print(f"COMBINED DATASET PROFILE ({label})")
    print("=" * 60)
    df = load_all(prefix=args.prefix, pv_type=args.pv_type)
    profile(df)
