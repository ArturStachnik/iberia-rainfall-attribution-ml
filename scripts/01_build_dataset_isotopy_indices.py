from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError("Repository root not found (missing .project_root).")


def _parse_monthly(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(series.astype(str).str.strip() + "-01", errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp(how="start")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dedup", choices=["first", "last"], default="first")
    args = parser.parse_args()

    ROOT = find_repo_root(Path(__file__))

    ISOTOPY_FILE = ROOT / "data" / "Segura_Isotopy_Monthly.csv"
    INDICES_FILE = ROOT / "data" / "WeMO_NAO_1950_2025.csv"
    OUT_FILE = ROOT / "data" / "Segura_Isotopy.csv"

    iso = pd.read_csv(ISOTOPY_FILE)
    idx = pd.read_csv(INDICES_FILE)

    if "Date" not in iso.columns:
        raise KeyError("Segura_Isotopy_Monthly must contain a 'Date' column.")
    if "Date" not in idx.columns:
        raise KeyError("WeMO_NAO_1950_2025 must contain a 'Date' column.")

    iso = iso.copy()
    idx = idx.copy()

    iso["Date"] = _parse_monthly(iso["Date"])
    idx["Date"] = _parse_monthly(idx["Date"])

    missing = [c for c in ["NAOi", "WeMOi"] if c not in idx.columns]
    if missing:
        raise KeyError(f"Indices file is missing columns: {missing}. Available: {list(idx.columns)}")

    idx_small = idx[["Date", "NAOi", "WeMOi"]].copy()
    idx_small = idx_small.sort_values("Date").drop_duplicates(subset=["Date"], keep=args.dedup)

    merged = iso.merge(idx_small, on="Date", how="left", validate="m:1")

    sort_cols = [c for c in ["Location", "Station", "Date"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    main()
