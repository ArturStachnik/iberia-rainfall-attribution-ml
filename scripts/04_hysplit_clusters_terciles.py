#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError("Repository root not found (missing .project_root).")


REPO_ROOT = find_repo_root(Path(__file__))

BASE = REPO_ROOT / "Isotopy_Analysis"
if not (BASE / "data").exists():
    BASE = REPO_ROOT

BASE_HYSPLIT = BASE / "data" / "hysplit"
WET_DIR = BASE_HYSPLIT / "WET"
DRY_DIR = BASE_HYSPLIT / "DRY"

OUT_DATA_DIR = BASE_HYSPLIT / "clustering_terciles"
OUT_FIG_DIR = BASE / "figures" / "hysplit"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

ASSIGN_PKL = OUT_DATA_DIR / "assign.pkl"
OUT_ASSIGN = OUT_DATA_DIR / "trajectory_cluster_assignments.csv"
OUT_CENTROIDS = OUT_DATA_DIR / "cluster_centroids.csv"
OUT_META = OUT_DATA_DIR / "cluster_meta.csv"
OUT_META_SUPP = OUT_DATA_DIR / "cluster_meta_supp_table.csv"

EVENT_DAYS_CANDIDATES = [
    BASE_HYSPLIT / "hysplit_event_days_common.csv",
    OUT_DATA_DIR / "hysplit_event_days_common.csv",
]

IDX_CSV = BASE / "data" / "WeMO_NAO_1950_2025.csv"

ALTITUDES = [500, 1500, 3000]
SEASONS = ["WET", "DRY"]
LAGS_HOURS = [0, 24, 48, 72, 96]

K_WET_REQ = 4
K_DRY_REQ = 3
MIN_CLUSTER_SIZE = 3

MAP_EXTENT = (-60, 25, 20, 65)
TERC_LABELS = ["low", "neutral", "high"]

# If you already have OUT_DATA_DIR/assign.pkl created by this script, keep False.
RECOMPUTE_CLUSTERING = False

# Spaghetti only in CLUSTERS_PANEL
SHOW_SPAGHETTI_IN_CLUSTERS_PANEL = True
SPAGHETTI_LW = 0.35
SPAGHETTI_ALPHA = 0.08

# Nature-ish defaults
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
})

USE_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.ticker as mticker
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    USE_CARTOPY = False


FNAME_RE = re.compile(
    r"(?P<prefix>Segura)(?P<mon>[a-z]{3})(?P<alt>\d{4})(?P<seasonword>[a-z]+)(?P<yyyymmdd>\d{8})(?P<hh>\d{2})$",
    re.IGNORECASE
)


def parse_from_filename(fname: str):
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = FNAME_RE.match(stem)
    if not m:
        return pd.NaT, np.nan
    alt = int(m.group("alt"))
    yyyymmdd = m.group("yyyymmdd")
    hh = int(m.group("hh"))
    dt = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
    if pd.notnull(dt):
        dt = dt + pd.Timedelta(hours=hh)
    return dt, alt


def read_hysplit_file(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                vals = [float(x) for x in parts[:13]]
            except Exception:
                continue
            yy = int(vals[2]); mm = int(vals[3]); dd = int(vals[4]); hh = int(vals[5])
            age_h = float(vals[8])
            lat = float(vals[9]); lon = float(vals[10]); height = float(vals[11])
            year = 2000 + yy if yy <= 50 else 1900 + yy
            dt = pd.Timestamp(year=year, month=mm, day=dd, hour=hh)
            lag_h = int(abs(int(round(age_h))))
            rows.append((dt, age_h, lag_h, lat, lon, height))
    if not rows:
        raise ValueError(f"No trajectory rows parsed: {path}")
    df = pd.DataFrame(rows, columns=["datetime", "age_h", "lag_h", "lat", "lon", "height_m"])
    df = df.sort_values("lag_h").drop_duplicates("lag_h", keep="first").reset_index(drop=True)
    return df


def sample_lonlat_at_lags(df: pd.DataFrame, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lagvals = df["lag_h"].to_numpy()
    lonvals = df["lon"].to_numpy()
    latvals = df["lat"].to_numpy()
    out_lon, out_lat = [], []
    for L in lags:
        idx = int(np.argmin(np.abs(lagvals - L)))
        out_lon.append(float(lonvals[idx]))
        out_lat.append(float(latvals[idx]))
    return np.array(out_lon), np.array(out_lat)


def build_feature_vector(traj_df: pd.DataFrame, lags: list[int]) -> np.ndarray:
    lon, lat = sample_lonlat_at_lags(traj_df, lags)
    x = np.empty(len(lags) * 2, dtype=float)
    x[0::2] = lon
    x[1::2] = lat
    return x


def kmeans_simple(X, k, n_init=40, max_iter=400, seed=42):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    n = len(X)
    best_inertia = np.inf
    best_labels, best_C = None, None

    for _ in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        C = X[idx].copy()
        labels = np.zeros(n, dtype=int)

        for _it in range(max_iter):
            dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            for j in range(k):
                if np.any(labels == j):
                    C[j] = X[labels == j].mean(axis=0)
                else:
                    C[j] = X[rng.integers(0, n)]

        inertia = ((X - C[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels, best_C = labels.copy(), C.copy()

    return best_labels, best_C


def silhouette_score_fast(X, labels):
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    n = len(X)
    if n < 3 or len(np.unique(labels)) < 2:
        return np.nan
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    svals = []
    for i in range(n):
        same = labels == labels[i]
        other = labels != labels[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = np.inf
        for lab in np.unique(labels[other]):
            mask = labels == lab
            b = min(b, D[i, mask].mean())
        svals.append((b - a) / max(a, b))
    return float(np.nanmean(svals)) if len(svals) else np.nan


def merge_tiny_clusters(X, labels, min_size=3):
    X = np.asarray(X, float)
    labels = np.asarray(labels).copy()

    def centroids_for(lbls):
        return {int(cl): X[lbls == cl].mean(axis=0) for cl in np.unique(lbls)}

    changed = True
    while changed:
        changed = False
        counts = pd.Series(labels).value_counts()
        tiny = [int(cl) for cl, n in counts.items() if n < min_size]
        if not tiny:
            break
        cents = centroids_for(labels)
        for tcl in tiny:
            counts = pd.Series(labels).value_counts()
            candidates = [int(c) for c, n in counts.items() if int(c) != tcl and n >= min_size]
            if not candidates:
                candidates = [int(counts.drop(index=tcl).idxmax())]
            ct = cents[tcl]
            best, best_d = None, np.inf
            for c in candidates:
                d = np.sum((ct - cents[c]) ** 2)
                if d < best_d:
                    best_d = d
                    best = c
            labels[labels == tcl] = best
            changed = True

    uniq = sorted(np.unique(labels))
    mapping = {old: new for new, old in enumerate(uniq)}
    labels = np.array([mapping[int(x)] for x in labels], dtype=int)
    return labels


def collect_files(folder: Path) -> list[str]:
    allf = [p for p in glob.glob(str(folder / "Segura*")) if os.path.isfile(p)]
    return sorted(allf)


def _safe_ts(x):
    return pd.to_datetime(x, errors="coerce")


def _safe_day(x):
    ts = _safe_ts(x)
    return pd.NaT if pd.isna(ts) else ts.normalize()


def _safe_month_start(x):
    ts = _safe_ts(x)
    return pd.NaT if pd.isna(ts) else pd.Timestamp(ts.year, ts.month, 1)


def quantile_tercile(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1, q2 = s.quantile([1/3, 2/3])
    out = pd.Series(index=s.index, dtype="object")
    out[s <= q1] = "low"
    out[(s > q1) & (s <= q2)] = "neutral"
    out[s > q2] = "high"
    return out


def composition_within_terciles(df: pd.DataFrame, cluster_col: str, terc_col: str, terc_labels: list[str]) -> pd.DataFrame:
    tmp = df[[cluster_col, terc_col]].dropna().copy()
    tmp[cluster_col] = tmp[cluster_col].astype(int)
    ct = pd.crosstab(tmp[terc_col], tmp[cluster_col]).reindex(terc_labels).fillna(0)
    pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    return pct.fillna(0)


def cluster_colors_for_panel(clusters: list[int]):
    cmap = plt.cm.magma
    m = max(clusters) if len(clusters) else 0
    return {cl: cmap((cl + 1) / (m + 2)) for cl in clusters}


def _detect_event_days_csv() -> Path:
    for p in EVENT_DAYS_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("hysplit_event_days_common.csv not found in expected locations.")


def _load_event_days(path: Path) -> set[pd.Timestamp]:
    ev = pd.read_csv(path)
    date_candidates = [c for c in ev.columns if str(c).lower() in ("date", "day", "datetime", "event_day", "event_date")]
    if not date_candidates:
        date_candidates = [c for c in ev.columns if ("date" in str(c).lower() or "day" in str(c).lower())]
    if not date_candidates:
        raise ValueError(f"No date-like column found in: {path}")
    col = date_candidates[0]
    ev["event_day"] = pd.to_datetime(ev[col], errors="coerce").dt.normalize()
    return set(ev["event_day"].dropna().unique())


def _parse_dates_robust(s: pd.Series) -> pd.Series:
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if dt1.notna().mean() >= 0.95:
        return dt1
    dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    if dt2.notna().mean() >= dt1.notna().mean():
        return dt2
    return dt1


def _load_monthly_indices(idx_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(idx_csv)
    date_candidates = [c for c in df.columns if str(c).lower() in ("date", "datetime", "time", "month", "month_start")]
    if not date_candidates:
        date_candidates = [c for c in df.columns if ("date" in str(c).lower() or "time" in str(c).lower() or "month" in str(c).lower())]
    if not date_candidates:
        raise ValueError("Could not detect date column in WeMO_NAO_1950_2025.csv")
    date_col = date_candidates[0]

    def pick_col(keys):
        for c in df.columns:
            lc = str(c).lower()
            if any(k in lc for k in keys):
                return c
        return None

    nao_col = pick_col(["nao"])
    wemo_col = pick_col(["wemo"])
    if nao_col is None or wemo_col is None:
        raise ValueError("Could not detect NAO/WeMO columns in WeMO_NAO_1950_2025.csv")

    df[date_col] = _parse_dates_robust(df[date_col])
    df["month_start"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month_start")[[nao_col, wemo_col]]
          .median()
          .reset_index()
          .rename(columns={nao_col: "NAO", wemo_col: "WeMO"})
    )
    return monthly


def _save_figure(fig: plt.Figure, basepath: Path, dpi: int = 600):
    fig.savefig(str(basepath.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(basepath.with_suffix(".svg")), bbox_inches="tight")


def _setup_gridliner(ax, show_left: bool, show_bottom: bool):
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = bool(show_left)
    gl.bottom_labels = bool(show_bottom)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.xlocator = mticker.FixedLocator([-60, -40, -20, 0, 20])
    gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60])
    return gl


def _setup_map_ax(fig, nrows, ncols, idx, show_labels_left: bool, show_labels_bottom: bool):
    if USE_CARTOPY and HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(nrows, ncols, idx, projection=proj)
        ax.set_extent(MAP_EXTENT, crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.15)
        _setup_gridliner(ax, show_labels_left, show_labels_bottom)
        return ax, proj

    ax = fig.add_subplot(nrows, ncols, idx)
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.grid(True, alpha=0.2)
    if show_labels_left:
        ax.set_ylabel("Latitude")
    if show_labels_bottom:
        ax.set_xlabel("Longitude")
    return ax, None


def _plot_clusters_grid(assign: pd.DataFrame, out_base: Path):
    fig = plt.figure(figsize=(18, 9))

    for i, season in enumerate(SEASONS):
        for j, alt in enumerate(ALTITUDES):
            idx = i * len(ALTITUDES) + j + 1
            show_left = (j == 0)
            # Requested: longitude ticks/labels also under the TOP row panels
            show_bottom = True
            ax, proj = _setup_map_ax(fig, 2, 3, idx, show_left, show_bottom)

            sub = assign[
                (assign["season"] == season) &
                (assign["start_alt_m"].astype(int) == int(alt)) &
                (assign["cluster"].astype(int) >= 0)
            ].copy()

            if len(sub) == 0:
                ax.set_title(f"{season} | {alt} m | N=0")
                continue

            clusters = sorted(sub["cluster"].astype(int).unique())
            colors = cluster_colors_for_panel(clusters)

            # Spaghetti
            if SHOW_SPAGHETTI_IN_CLUSTERS_PANEL:
                for _, r in sub.iterrows():
                    df = r["traj_df"]
                    lon = df["lon"].to_numpy()
                    lat = df["lat"].to_numpy()
                    col = colors[int(r["cluster"])]
                    if proj is not None:
                        ax.plot(lon, lat, linewidth=SPAGHETTI_LW, alpha=SPAGHETTI_ALPHA, color=col, transform=ccrs.PlateCarree())
                    else:
                        ax.plot(lon, lat, linewidth=SPAGHETTI_LW, alpha=SPAGHETTI_ALPHA, color=col)

            # Means (cluster mean back trajectories)
            for cl in clusters:
                g = sub[sub["cluster"].astype(int) == cl]
                LON, LAT = [], []
                for _, r in g.iterrows():
                    lon_s, lat_s = sample_lonlat_at_lags(r["traj_df"], LAGS_HOURS)
                    LON.append(lon_s); LAT.append(lat_s)
                m_lon = np.nanmean(np.vstack(LON), axis=0)
                m_lat = np.nanmean(np.vstack(LAT), axis=0)
                if proj is not None:
                    ax.plot(m_lon, m_lat, linewidth=3.2, alpha=0.95, color=colors[cl], transform=ccrs.PlateCarree())
                else:
                    ax.plot(m_lon, m_lat, linewidth=3.2, alpha=0.95, color=colors[cl])

            ax.set_title(f"{season} | {alt} m | N={len(sub)} | K={len(clusters)}")

    plt.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def _plot_terciles_grid(assign: pd.DataFrame, ev_days: set[pd.Timestamp], monthly_idx: pd.DataFrame, out_base: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    for i, season in enumerate(SEASONS):
        for j, alt in enumerate(ALTITUDES):
            ax = axes[i, j]
            sub = assign[
                (assign["season"] == season) &
                (assign["start_alt_m"].astype(int) == int(alt)) &
                (assign["cluster"].astype(int) >= 0) &
                (assign["event_day"].isin(ev_days))
            ].copy()

            if len(sub) == 0:
                ax.set_title(f"{season} | {alt} m | N=0")
                ax.axis("off")
                continue

            sub = sub.merge(monthly_idx, on="month_start", how="left")
            sub = sub.dropna(subset=["NAO", "WeMO"]).copy()
            if len(sub) == 0:
                ax.set_title(f"{season} | {alt} m | N=0")
                ax.axis("off")
                continue

            sub["NAO_terc"] = quantile_tercile(sub["NAO"])
            sub["WeMO_terc"] = quantile_tercile(sub["WeMO"])

            clusters = sorted(sub["cluster"].astype(int).unique())
            colors = cluster_colors_for_panel(clusters)

            tbl_nao = composition_within_terciles(sub, "cluster", "NAO_terc", TERC_LABELS)
            tbl_wemo = composition_within_terciles(sub, "cluster", "WeMO_terc", TERC_LABELS)

            x_nao = np.array([0, 1, 2], dtype=float)
            x_wemo = np.array([4, 5, 6], dtype=float)

            bottom = np.zeros(3)
            for cl in clusters:
                vals = tbl_nao.reindex(TERC_LABELS)[cl].to_numpy() if cl in tbl_nao.columns else np.zeros(3)
                ax.bar(x_nao, vals, bottom=bottom, color=colors[cl], edgecolor="white", linewidth=0.6)
                bottom += vals

            bottom = np.zeros(3)
            for cl in clusters:
                vals = tbl_wemo.reindex(TERC_LABELS)[cl].to_numpy() if cl in tbl_wemo.columns else np.zeros(3)
                ax.bar(x_wemo, vals, bottom=bottom, color=colors[cl], edgecolor="white", linewidth=0.6)
                bottom += vals

            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([0, 1, 2, 4, 5, 6])
            ax.set_xticklabels(["NAO-low", "NAO-neu", "NAO-high", "WeMO-low", "WeMO-neu", "WeMO-high"], fontsize=9)
            if j == 0:
                ax.set_ylabel("% events (within tercile)")
            ax.set_title(f"{season} | {alt} m | N={len(sub)} | K={len(clusters)}", fontsize=10)

    _save_figure(fig, out_base)
    plt.close(fig)


def _plot_combined_panels(assign: pd.DataFrame, ev_days: set[pd.Timestamp], monthly_idx: pd.DataFrame):
    for season in SEASONS:
        for alt in ALTITUDES:
            sub = assign[
                (assign["season"] == season) &
                (assign["start_alt_m"].astype(int) == int(alt)) &
                (assign["cluster"].astype(int) >= 0) &
                (assign["event_day"].isin(ev_days))
            ].copy()

            if len(sub) == 0:
                continue

            sub = sub.merge(monthly_idx, on="month_start", how="left")
            sub = sub.dropna(subset=["NAO", "WeMO"]).copy()
            if len(sub) == 0:
                continue

            sub["NAO_terc"] = quantile_tercile(sub["NAO"])
            sub["WeMO_terc"] = quantile_tercile(sub["WeMO"])

            clusters = sorted(sub["cluster"].astype(int).unique())
            colors = cluster_colors_for_panel(clusters)

            tbl_nao = composition_within_terciles(sub, "cluster", "NAO_terc", TERC_LABELS)
            tbl_wemo = composition_within_terciles(sub, "cluster", "WeMO_terc", TERC_LABELS)

            fig = plt.figure(figsize=(18, 6.5))
            gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[2.8, 1.0], height_ratios=[1, 1], wspace=0.18, hspace=0.35)

            if USE_CARTOPY and HAS_CARTOPY:
                proj = ccrs.PlateCarree()
                ax_map = fig.add_subplot(gs[:, 0], projection=proj)
                ax_map.set_extent(MAP_EXTENT, crs=proj)
                ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7)
                ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_map.add_feature(cfeature.LAND, alpha=0.15)
                _setup_gridliner(ax_map, show_left=True, show_bottom=True)
            else:
                ax_map = fig.add_subplot(gs[:, 0])
                ax_map.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
                ax_map.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
                ax_map.grid(True, alpha=0.2)
                ax_map.set_xlabel("Longitude")
                ax_map.set_ylabel("Latitude")

            ax_nao = fig.add_subplot(gs[0, 1])
            ax_wemo = fig.add_subplot(gs[1, 1])

            for cl in clusters:
                g = sub[sub["cluster"].astype(int) == cl]
                LON, LAT = [], []
                for _, r in g.iterrows():
                    lon_s, lat_s = sample_lonlat_at_lags(r["traj_df"], LAGS_HOURS)
                    LON.append(lon_s); LAT.append(lat_s)
                m_lon = np.nanmean(np.vstack(LON), axis=0)
                m_lat = np.nanmean(np.vstack(LAT), axis=0)
                if USE_CARTOPY and HAS_CARTOPY:
                    ax_map.plot(m_lon, m_lat, linewidth=3.5, alpha=0.95, color=colors[cl], transform=ccrs.PlateCarree())
                else:
                    ax_map.plot(m_lon, m_lat, linewidth=3.5, alpha=0.95, color=colors[cl])

            ax_map.set_title(f"Cluster mean Back Trajectories (N={len(sub)})", fontsize=11)

            for ax, title, tbl in [(ax_nao, "NAO terciles", tbl_nao), (ax_wemo, "WeMO terciles", tbl_wemo)]:
                ax.set_title(title, fontsize=10)
                ax.set_ylim(0, 100)
                ax.grid(axis="y", alpha=0.25)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_ylabel("% events")
                x = np.array([0, 1, 2], dtype=float)
                bottom = np.zeros(3)
                for cl in clusters:
                    vals = tbl.reindex(TERC_LABELS)[cl].to_numpy() if cl in tbl.columns else np.zeros(3)
                    ax.bar(x, vals, bottom=bottom, color=colors[cl], edgecolor="white", linewidth=0.6)
                    bottom += vals
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(TERC_LABELS)

            out_base = OUT_FIG_DIR / f"{season}_{alt}m"
            _save_figure(fig, out_base)
            plt.close(fig)


def run_clustering() -> pd.DataFrame:
    wet_files = collect_files(WET_DIR)
    dry_files = collect_files(DRY_DIR)
    if len(wet_files) == 0 and len(dry_files) == 0:
        raise SystemExit("No trajectory files found in Isotopy_Analysis/data/hysplit/WET or DRY.")

    rows = []
    for season, files in [("WET", wet_files), ("DRY", dry_files)]:
        for fp in files:
            fname = os.path.basename(fp)
            dt, alt = parse_from_filename(fname)
            if pd.isna(dt) or (not np.isfinite(alt)) or (int(alt) not in ALTITUDES):
                continue
            try:
                tdf = read_hysplit_file(fp)
            except Exception:
                continue
            rows.append({
                "season": season,
                "traj_file": fname,
                "path": fp,
                "datetime_utc": dt,
                "date": dt.normalize(),
                "start_alt_m": int(alt),
                "traj_df": tdf,
            })

    assign = pd.DataFrame(rows)
    if len(assign) == 0:
        raise SystemExit("Parsed 0 trajectories. Check filename pattern and file format.")

    assign["cluster"] = -1
    meta = []

    for season in SEASONS:
        K_req = K_WET_REQ if season == "WET" else K_DRY_REQ
        for alt in ALTITUDES:
            sub = assign[(assign["season"] == season) & (assign["start_alt_m"] == alt)].copy()
            n = len(sub)
            if n < max(3, K_req):
                continue
            Xsub = np.vstack([build_feature_vector(r["traj_df"], LAGS_HOURS) for _, r in sub.iterrows()])
            labels, _ = kmeans_simple(Xsub, K_req)
            labels = merge_tiny_clusters(Xsub, labels, min_size=MIN_CLUSTER_SIZE)

            counts = pd.Series(labels).value_counts().sort_index()
            K_final = int(len(counts))
            sil = silhouette_score_fast(Xsub, labels)

            assign.loc[sub.index, "cluster"] = labels.astype(int)

            sizes_str = ";".join([f"{int(k)}:{int(v)}" for k, v in counts.items()])
            meta.append({
                "season": season,
                "start_alt_m": int(alt),
                "K_requested": int(K_req),
                "min_cluster_size": int(MIN_CLUSTER_SIZE),
                "K_final": int(K_final),
                "silhouette": float(sil) if sil == sil else np.nan,
                "n": int(n),
                "cluster_sizes": sizes_str,
            })

    meta_df = pd.DataFrame(meta).sort_values(["season", "start_alt_m"]).reset_index(drop=True)
    meta_df.to_csv(OUT_META, index=False)

    meta_supp = meta_df.copy().rename(columns={"start_alt_m": "arrival_height_m", "n": "n_events"})
    meta_supp["silhouette"] = pd.to_numeric(meta_supp["silhouette"], errors="coerce").round(3)
    meta_supp.to_csv(OUT_META_SUPP, index=False)

    cent_rows = []
    for season in SEASONS:
        for alt in ALTITUDES:
            sub = assign[(assign["season"] == season) & (assign["start_alt_m"] == alt) & (assign["cluster"] >= 0)]
            if len(sub) == 0:
                continue
            for cl in sorted(sub["cluster"].astype(int).unique()):
                g = sub[sub["cluster"].astype(int) == cl]
                LON, LAT = [], []
                for _, r in g.iterrows():
                    lon, lat = sample_lonlat_at_lags(r["traj_df"], LAGS_HOURS)
                    LON.append(lon); LAT.append(lat)
                m_lon = np.nanmean(np.vstack(LON), axis=0)
                m_lat = np.nanmean(np.vstack(LAT), axis=0)
                for L, lo, la in zip(LAGS_HOURS, m_lon, m_lat):
                    cent_rows.append({
                        "season": season,
                        "start_alt_m": int(alt),
                        "cluster": int(cl),
                        "lag_h": int(L),
                        "mean_lon": float(lo),
                        "mean_lat": float(la),
                        "n_traj": int(len(g)),
                    })
    pd.DataFrame(cent_rows).to_csv(OUT_CENTROIDS, index=False)

    assign.drop(columns=["traj_df"]).to_csv(OUT_ASSIGN, index=False)
    assign.to_pickle(ASSIGN_PKL)

    return assign


def main():
    if RECOMPUTE_CLUSTERING or (not ASSIGN_PKL.exists()):
        assign = run_clustering()
    else:
        assign = pd.read_pickle(ASSIGN_PKL)

    assign["datetime_utc"] = pd.to_datetime(assign["datetime_utc"], errors="coerce")
    assign["event_day"] = assign["datetime_utc"].apply(_safe_day)
    assign["month_start"] = assign["datetime_utc"].apply(_safe_month_start)
    assign["start_alt_m"] = pd.to_numeric(assign["start_alt_m"], errors="coerce").astype("Int64")
    assign["cluster"] = pd.to_numeric(assign["cluster"], errors="coerce").astype("Int64")

    ev_csv = _detect_event_days_csv()
    ev_days = _load_event_days(ev_csv)
    monthly_idx = _load_monthly_indices(IDX_CSV)

    _plot_clusters_grid(assign, OUT_FIG_DIR / "CLUSTERS_PANEL")
    _plot_terciles_grid(assign, ev_days, monthly_idx, OUT_FIG_DIR / "TERCILES_PANEL")
    _plot_combined_panels(assign, ev_days, monthly_idx)


if __name__ == "__main__":
    main()
