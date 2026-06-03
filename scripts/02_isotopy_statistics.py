from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# Global style (paper-like)
# =========================
DPI = 300
CMAP = "coolwarm"
VMIN, VMAX = -1.0, 1.0


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError("Repository root not found (missing .project_root).")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_both(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")


def _parse_monthly(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(s + "-01", errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp(how="start")


WET_MONTHS = [10, 11, 12, 1, 2, 3]   # ONDJFM
DRY_MONTHS = [4, 5, 6, 7, 8, 9]      # AMJJAS


def _add_season(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["Date"].dt.month
    out["Season"] = np.where(
        out["month"].isin(WET_MONTHS),
        "WET",
        np.where(out["month"].isin(DRY_MONTHS), "DRY", "OTHER"),
    )
    return out


def _annot_color(val: float, thresh: float = 0.60) -> str:
    # white numbers when strong |rho|
    if not np.isfinite(val):
        return "black"
    return "white" if abs(val) >= thresh else "black"


def _heatmap_with_dynamic_annot(
    ax: plt.Axes,
    mat: pd.DataFrame,
    *,
    title: str,
    cbar: bool,
    cbar_ax: plt.Axes | None,
    cbar_label: str | None,
    fmt: str = ".2f",
    annot_thresh: float = 0.60,
) -> None:
    sns.heatmap(
        mat,
        ax=ax,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        square=True,
        cbar=cbar,
        cbar_ax=cbar_ax,
        annot=False,  # we add our own (to control white/black)
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", length=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat.values[i, j])
            if np.isfinite(v):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    format(v, fmt),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_annot_color(v, thresh=annot_thresh),
                )

    if cbar_ax is not None and cbar_label is not None:
        cbar_ax.set_ylabel(cbar_label, fontsize=10)


def fig_spearman_wet_dry(df: pd.DataFrame, out_base: Path) -> None:
    candidate_vars = [
        "d18O", "dH", "d17O", "O17ex", "dex",
        "pcp_sum",
        "Altitude", "Latitude", "Longitude",
        "NAOi", "WeMOi",
    ]
    # ensure numeric
    for c in candidate_vars:
        if c in df.columns and c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    vars_for_corr = [v for v in candidate_vars if v in df.columns]

    wet = df[df["Season"] == "WET"][vars_for_corr].corr(method="spearman")
    dry = df[df["Season"] == "DRY"][vars_for_corr].corr(method="spearman")

    sns.set(style="white", font_scale=0.9)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # put colorbar OUTSIDE (no tight_layout here)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])

    _heatmap_with_dynamic_annot(
        axes[0], wet,
        title="A) Wet season (ONDJFM)",
        cbar=True, cbar_ax=cbar_ax, cbar_label="Spearman ρ",
        annot_thresh=0.60,
    )
    _heatmap_with_dynamic_annot(
        axes[1], dry,
        title="B) Dry season (AMJJAS)",
        cbar=False, cbar_ax=None, cbar_label=None,
        annot_thresh=0.60,
    )

    plt.subplots_adjust(right=0.9, wspace=0.25)
    _save_both(fig, out_base)
    plt.close(fig)


def _run_pca(df_season: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    data = df_season[features].dropna()
    if len(data) < len(features):
        raise ValueError(f"Not enough rows for PCA: {len(data)} rows, need >= {len(features)}.")

    X = StandardScaler().fit_transform(data.values)
    pca = PCA(n_components=len(features))
    pca.fit(X)

    exp = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))],
    )
    return exp, loadings


def _force_pca_sign(loadings: pd.DataFrame, *, pc: str, anchor: str, want_negative: bool = True) -> pd.DataFrame:
    """
    PCA sign is arbitrary. To reproduce the paper exactly, enforce a convention:
    e.g., NAOi loading on PC1 must be negative. If not, flip PC1.
    """
    L = loadings.copy()
    if pc not in L.columns or anchor not in L.index:
        return L
    val = float(L.loc[anchor, pc])
    if want_negative and val > 0:
        L[pc] = -L[pc]
    if (not want_negative) and val < 0:
        L[pc] = -L[pc]
    return L


def fig_pca_panel(df: pd.DataFrame, out_base: Path) -> None:
    FEATURES = ["NAOi", "WeMOi", "pcp_sum", "Altitude"]

    df = df.copy()
    df["Date"] = _parse_monthly(df["Date"])
    df = _add_season(df)

    df_wet = df[df["Season"] == "WET"][FEATURES].dropna()
    df_dry = df[df["Season"] == "DRY"][FEATURES].dropna()

    exp_wet, load_wet = _run_pca(df_wet, FEATURES)
    exp_dry, load_dry = _run_pca(df_dry, FEATURES)

    def _flip_pc(load: pd.DataFrame, pc: str) -> pd.DataFrame:
        out = load.copy()
        out[pc] = -out[pc]
        return out

    if "NAOi" in load_wet.index and load_wet.loc["NAOi", "PC1"] > 0:
        load_wet = _flip_pc(load_wet, "PC1")
    if "pcp_sum" in load_wet.index and load_wet.loc["pcp_sum", "PC2"] < 0:
        load_wet = _flip_pc(load_wet, "PC2")

    if "NAOi" in load_dry.index and load_dry.loc["NAOi", "PC1"] < 0:
        load_dry = _flip_pc(load_dry, "PC1")

    out_dir = out_base.parent
    _ensure_dir(out_dir)

    load_wet.to_csv(out_dir / "pca_loadings_wet.csv", index=True)
    load_dry.to_csv(out_dir / "pca_loadings_dry.csv", index=True)

    exp_df = pd.DataFrame(
        {
            "Season": ["WET", "DRY"],
            "PC1": [exp_wet[0], exp_dry[0]],
            "PC2": [exp_wet[1], exp_dry[1]],
            "PC3": [exp_wet[2], exp_dry[2]],
            "PC4": [exp_wet[3], exp_dry[3]],
        }
    )
    exp_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)

    def _plot(ax: plt.Axes, load: pd.DataFrame, exp: np.ndarray, pcx: str, pcy: str, title: str) -> None:
        ax.set_axisbelow(True)
        ax.axhline(0, color="0.80", lw=1.0, zorder=0)
        ax.axvline(0, color="0.80", lw=1.0, zorder=0)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", adjustable="box")

        ix = int(pcx.replace("PC", "")) - 1
        iy = int(pcy.replace("PC", "")) - 1
        ax.set_xlabel(f"{pcx} ({exp[ix]*100:.1f}%)")
        ax.set_ylabel(f"{pcy} ({exp[iy]*100:.1f}%)")
        ax.set_title(title)

        for feat in load.index:
            x = float(load.loc[feat, pcx])
            y = float(load.loc[feat, pcy])
            ax.arrow(0, 0, x, y, color="red", head_width=0.045, length_includes_head=True, zorder=3)
            ax.text(x * 1.10, y * 1.10, feat, color="red", ha="center", va="center", fontsize=10, zorder=4)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))

    _plot(axes[0, 0], load_wet, exp_wet, "PC1", "PC2", "WET season (ONDJFM) — PC1 vs PC2")
    _plot(axes[0, 1], load_dry, exp_dry, "PC1", "PC2", "DRY season (AMJJAS) — PC1 vs PC2")
    _plot(axes[1, 0], load_wet, exp_wet, "PC1", "PC3", "WET season (ONDJFM) — PC1 vs PC3")
    _plot(axes[1, 1], load_dry, exp_dry, "PC1", "PC3", "DRY season (AMJJAS) — PC1 vs PC3")

    fig.tight_layout()
    _save_both(fig, out_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Segura_Isotopy.csv")
    args = parser.parse_args()

    root = find_repo_root(Path(__file__))
    data_path = (root / args.data).resolve()

    df = pd.read_csv(data_path)
    df["Date"] = _parse_monthly(df["Date"])
    df = _add_season(df)

    figures_dir = root / "figures"
    corr_dir = figures_dir / "correlations"
    pca_dir = figures_dir / "pca"
    _ensure_dir(corr_dir)
    _ensure_dir(pca_dir)

    fig_spearman_wet_dry(df, corr_dir / "spearman_correlation_matrices_wet_dry")
    fig_pca_panel(df, pca_dir / "vectors" / "pca_vectors_panel")


if __name__ == "__main__":
    main()
