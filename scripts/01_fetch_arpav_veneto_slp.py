from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from pandas.tseries.offsets import MonthBegin


VENETO_STATIONS: Dict[str, str] = {
    "Malo": "300000828",
    "Teolo": "300001259",
    "Venezia": "300005146",
    "Cavallino": "300001101",
}

VENETO_BASE_URL = "https://api.arpa.veneto.it/REST/v1/meteo_storici_tabella"

START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp("2025-12-31")

REQUEST_TIMEOUT_S = 40
MAX_RETRIES = 6
BACKOFF_BASE_S = 1.25


@dataclass(frozen=True)
class VenetoRequestSpec:
    codseq: str
    year: int


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError("Repository root not found (missing .project_root).")


ROOT = find_repo_root(Path(__file__))
OUT_DIR = ROOT / "data" / "ARPAV"
OUT_CSV = OUT_DIR / "veneto_slp_monthly_2010_2025.csv"


def month_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start_ms = pd.Timestamp(start.date()) + MonthBegin(0)
    end_ms = pd.Timestamp(end.date()) + MonthBegin(0)
    return pd.date_range(start_ms, end_ms, freq="MS")


def request_json(session: requests.Session, url: str, params: dict) -> Optional[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception:
            sleep_s = BACKOFF_BASE_S ** attempt + 0.15 * attempt
            time.sleep(min(sleep_s, 30.0))
    return None


def fetch_daily_pressure_year(session: requests.Session, spec: VenetoRequestSpec) -> pd.DataFrame:
    payload = request_json(session, VENETO_BASE_URL, {"codseq": spec.codseq, "anno": spec.year})
    if not payload or not payload.get("success", False):
        return pd.DataFrame(columns=["datetime_utc", "slp_hpa"])

    records: List[Tuple[pd.Timestamp, Optional[float]]] = []

    for item in payload.get("data", []) or []:
        if item.get("tipo") != "PRESS":
            continue

        ts = pd.to_datetime(item.get("dataora"), errors="coerce", utc=True)
        if pd.isna(ts):
            continue

        raw_val = item.get("valore")
        if raw_val is None:
            continue

        try:
            val = json.loads(raw_val)
        except Exception:
            continue

        slp = val.get("MEDIO", None)
        try:
            slp_f = float(slp) if slp is not None else None
        except Exception:
            slp_f = None

        records.append((ts, slp_f))

    if not records:
        return pd.DataFrame(columns=["datetime_utc", "slp_hpa"])

    df = pd.DataFrame(records, columns=["datetime_utc", "slp_hpa"])
    return df.sort_values("datetime_utc")


def daily_to_monthly(df_daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame(columns=["Date", "SLP"])

    d = df_daily.copy()
    d["datetime_utc"] = pd.to_datetime(d["datetime_utc"], utc=True, errors="coerce")
    d = d.dropna(subset=["datetime_utc"])
    d = d.set_index(d["datetime_utc"].dt.tz_convert(None)).drop(columns=["datetime_utc"])

    monthly = (
        d.resample("MS")["slp_hpa"]
        .mean()
        .rename("SLP")
        .to_frame()
        .reset_index()
    )

    monthly["Date"] = pd.to_datetime(monthly["datetime_utc"], errors="coerce")
    monthly = monthly[["Date", "SLP"]]

    start_ms = pd.Timestamp(start.date()) + MonthBegin(0)
    end_ms = pd.Timestamp(end.date()) + MonthBegin(0)
    monthly = monthly[(monthly["Date"] >= start_ms) & (monthly["Date"] <= end_ms)]

    return monthly


def build_veneto_monthly_dataset() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"accept": "application/json"})

    full_index = pd.DataFrame({"Date": month_index(START_DATE, END_DATE)})

    frames: List[pd.DataFrame] = []

    for station, codseq in VENETO_STATIONS.items():
        yearly_frames: List[pd.DataFrame] = []
        for year in range(START_DATE.year, END_DATE.year + 1):
            df_year = fetch_daily_pressure_year(session, VenetoRequestSpec(codseq=codseq, year=year))
            if not df_year.empty:
                yearly_frames.append(df_year)

        if yearly_frames:
            daily = pd.concat(yearly_frames, ignore_index=True)
            monthly = daily_to_monthly(daily, START_DATE, END_DATE)
            monthly = full_index.merge(monthly, on="Date", how="left")
        else:
            monthly = full_index.copy()
            monthly["SLP"] = pd.NA

        monthly["station"] = station
        monthly["codseq"] = codseq
        frames.append(monthly)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["Date", "station"]).reset_index(drop=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    return out[["Date", "SLP", "station", "codseq"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_veneto_monthly_dataset()
    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
