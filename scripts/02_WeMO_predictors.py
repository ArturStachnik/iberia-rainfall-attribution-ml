from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import MonthBegin


START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp("2025-12-01")

VENETO_STATION_TO_COL = {
    "Malo": "SLP_Malo",
    "Teolo": "SLP_Teolo",
    "Venezia": "SLP_Venezia",
    "Cavallino": "SLP_Cavallino",
}


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError("Repository root not found (missing .project_root).")


ROOT = find_repo_root(Path(__file__))

VENETO_CSV = ROOT / "data" / "ARPAV" / "veneto_slp_monthly_2010_2025.csv"
SANFER_CSV = ROOT / "data" / "AEMET" / "sanfernando_slp_monthly_2010_2025.csv"

OUT_DIR = ROOT / "data" / "predictors"
OUT_RAW = OUT_DIR / "wemo_predictors_monthly_2010_2025_raw.csv"
OUT_PROCESSED = OUT_DIR / "wemo_predictors_monthly_2010_2025_processed.csv"


def month_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_ms = pd.Timestamp(start.date()) + MonthBegin(0)
    end_ms = pd.Timestamp(end.date()) + MonthBegin(0)
    return pd.DataFrame({"Date": pd.date_range(start_ms, end_ms, freq="MS")})


def load_veneto_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["SLP"] = pd.to_numeric(df["SLP"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df = df[df["station"].isin(VENETO_STATION_TO_COL.keys())].copy()
    df["slp_col"] = df["station"].map(VENETO_STATION_TO_COL)

    wide = df.pivot_table(index="Date", columns="slp_col", values="SLP", aggfunc="mean").reset_index()
    return wide


def load_sanfernando(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["SLP"] = pd.to_numeric(df["SLP"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df[["Date", "SLP"]].rename(columns={"SLP": "SLP_SanFernando"})


def add_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    m = df["Date"].dt.month.astype(int)
    theta = 2.0 * np.pi * (m - 1) / 12.0
    out = df.copy()
    out["month_sin"] = np.sin(theta)
    out["month_cos"] = np.cos(theta)
    return out


def build_predictors(veneto_wide: pd.DataFrame, sanfer: pd.DataFrame) -> pd.DataFrame:
    idx = month_index(START_DATE, END_DATE)
    base = idx.merge(veneto_wide, on="Date", how="left").merge(sanfer, on="Date", how="left")

    base["D_Malo"] = base["SLP_SanFernando"] - base["SLP_Malo"]
    base["D_Teolo"] = base["SLP_SanFernando"] - base["SLP_Teolo"]
    base["D_Venezia"] = base["SLP_SanFernando"] - base["SLP_Venezia"]
    base["D_Cavallino"] = base["SLP_SanFernando"] - base["SLP_Cavallino"]

    base = add_cyclic_month(base)

    ordered = [
        "Date",
        "SLP_SanFernando",
        "SLP_Malo",
        "SLP_Teolo",
        "SLP_Venezia",
        "SLP_Cavallino",
        "D_Malo",
        "D_Teolo",
        "D_Venezia",
        "D_Cavallino",
        "month_sin",
        "month_cos",
    ]

    out = base[ordered].sort_values("Date").reset_index(drop=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


def processed_complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "SLP_SanFernando",
        "SLP_Malo",
        "SLP_Teolo",
        "SLP_Venezia",
        "SLP_Cavallino",
        "month_sin",
        "month_cos",
    ]
    return df.dropna(subset=required).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    veneto_wide = load_veneto_wide(VENETO_CSV)
    sanfer = load_sanfernando(SANFER_CSV)

    predictors = build_predictors(veneto_wide, sanfer)
    predictors.to_csv(OUT_RAW, index=False)

    predictors_processed = processed_complete_cases(predictors)
    predictors_processed.to_csv(OUT_PROCESSED, index=False)


if __name__ == "__main__":
    main()
