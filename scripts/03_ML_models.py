from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import shap



def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError("Repository root not found (missing .project_root).")


ROOT = find_repo_root(Path(__file__))

INPUT_CSV = ROOT / "data" / "Segura_Isotopy.csv"
FEATURES = ["NAOi", "WeMOi", "pcp_sum", "Altitude"]

WET_MONTHS = [10, 11, 12, 1, 2, 3]
DRY_MONTHS = [4, 5, 6, 7, 8, 9]

N_SPLITS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_KW = dict(
    n_estimators=300,
    max_depth=6,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
)

TARGETS = ["d18O", "O17ex", "dex"]


def parse_monthly_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    iso_ymd = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    iso_ym = s.str.match(r"^\d{4}-\d{2}$")

    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if iso_ymd.any():
        out.loc[iso_ymd] = pd.to_datetime(s.loc[iso_ymd], format="%Y-%m-%d", errors="coerce")

    if iso_ym.any():
        out.loc[iso_ym] = pd.to_datetime(s.loc[iso_ym] + "-01", format="%Y-%m-%d", errors="coerce")

    other = ~(iso_ymd | iso_ym)
    if other.any():
        dt = pd.to_datetime(s.loc[other], dayfirst=True, errors="coerce")
        fallback = pd.to_datetime(s.loc[other] + "-01", errors="coerce")
        out.loc[other] = dt.fillna(fallback)

    return out.dt.to_period("M").dt.to_timestamp(how="start")


def season_from_month(m: int) -> str:
    if m in WET_MONTHS:
        return "WET"
    if m in DRY_MONTHS:
        return "DRY"
    return "OTHER"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass(frozen=True)
class MonteCarloMetrics:
    r2_mean: float
    r2_std: float
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    n_samples: int
    season_label: str
    target: str


def monte_carlo_rf_for_season(
    df_season: pd.DataFrame,
    season_label: str,
    target: str,
    features: List[str],
    n_splits: int = N_SPLITS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame, MonteCarloMetrics, pd.DataFrame, pd.DataFrame, np.ndarray]:
    sub = df_season.dropna(subset=[target] + features).copy()
    sub = sub.sort_values("Date")

    X_df = sub[features].copy()
    y = sub[target].values
    X = X_df.values

    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    r2_scores: List[float] = []
    mae_scores: List[float] = []
    rmse_scores: List[float] = []

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestRegressor(random_state=split_idx, **RF_KW)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        r2_scores.append(float(r2_score(y_test, y_pred)))
        mae_scores.append(float(mean_absolute_error(y_test, y_pred)))
        rmse_scores.append(rmse(y_test, y_pred))

    r2_mean, r2_std = float(np.mean(r2_scores)), float(np.std(r2_scores))
    mae_mean, mae_std = float(np.mean(mae_scores)), float(np.std(mae_scores))
    rmse_mean, rmse_std = float(np.mean(rmse_scores)), float(np.std(rmse_scores))

    metrics = MonteCarloMetrics(
        r2_mean=r2_mean,
        r2_std=r2_std,
        mae_mean=mae_mean,
        mae_std=mae_std,
        rmse_mean=rmse_mean,
        rmse_std=rmse_std,
        n_samples=int(len(sub)),
        season_label=season_label,
        target=target,
    )

    final_rf = RandomForestRegressor(random_state=random_state, **RF_KW)
    final_rf.fit(X, y)

    fi_df = pd.DataFrame({"feature": features, "importance": final_rf.feature_importances_}).sort_values(
        "importance", ascending=True
    )

    perm = permutation_importance(
        final_rf,
        X,
        y,
        scoring="r2",
        n_repeats=100,
        random_state=random_state,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {"feature": features, "perm_mean": perm.importances_mean, "perm_std": perm.importances_std}
    ).sort_values("perm_mean", ascending=True)

    return final_rf, fi_df, perm_df, metrics, sub, X_df, y


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: object, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_metrics_txt(path: Path, metrics: MonteCarloMetrics) -> None:
    lines = [
        f"target: {metrics.target}",
        f"season: {metrics.season_label}",
        f"n_samples: {metrics.n_samples}",
        f"monte_carlo_splits: {N_SPLITS}",
        f"test_size: {TEST_SIZE}",
        f"random_state: {RANDOM_STATE}",
        "",
        f"R2_mean: {metrics.r2_mean:.6f}",
        f"R2_std: {metrics.r2_std:.6f}",
        f"MAE_mean: {metrics.mae_mean:.6f}",
        f"MAE_std: {metrics.mae_std:.6f}",
        f"RMSE_mean: {metrics.rmse_mean:.6f}",
        f"RMSE_std: {metrics.rmse_std:.6f}",
        "",
        "rf_params:",
        f"  n_estimators: {RF_KW['n_estimators']}",
        f"  max_depth: {RF_KW['max_depth']}",
        f"  min_samples_split: {RF_KW['min_samples_split']}",
        f"  min_samples_leaf: {RF_KW['min_samples_leaf']}",
        f"  max_features: {RF_KW['max_features']}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_fig(fig: plt.Figure, out_png: Path, out_svg: Path, *, dpi: int = 300, bbox_tight: bool = False) -> None:
    ensure_dir(out_png.parent)
    ensure_dir(out_svg.parent)
    if bbox_tight:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_svg, bbox_inches="tight")
    else:
        fig.savefig(out_png, dpi=dpi)
        fig.savefig(out_svg)
    plt.close(fig)


def plot_fig5(
    target: str,
    fi_wet: pd.DataFrame,
    perm_wet: pd.DataFrame,
    metrics_wet: MonteCarloMetrics,
    fi_dry: pd.DataFrame,
    perm_dry: pd.DataFrame,
    metrics_dry: MonteCarloMetrics,
) -> plt.Figure:
    sns.set(style="whitegrid", font_scale=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.barh(fi_wet["feature"], fi_wet["importance"])
    ax1.set_title(
        f"WET – RF feature importance\n(impurity-based)\n"
        f"MC R² = {metrics_wet.r2_mean:.2f} ± {metrics_wet.r2_std:.2f}"
    )
    ax1.set_xlabel("Importance")
    ax1.set_ylabel("Feature")

    ax2.errorbar(perm_wet["perm_mean"], perm_wet["feature"], xerr=perm_wet["perm_std"], fmt="o")
    ax2.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax2.set_title("WET – permutation importance\n(ΔR² on full dataset)")
    ax2.set_xlabel("ΔR² when permuted")
    ax2.set_ylabel("")

    ax3.barh(fi_dry["feature"], fi_dry["importance"])
    ax3.set_title(
        f"DRY – RF feature importance\n(impurity-based)\n"
        f"MC R² = {metrics_dry.r2_mean:.2f} ± {metrics_dry.r2_std:.2f}"
    )
    ax3.set_xlabel("Importance")
    ax3.set_ylabel("Feature")

    ax4.errorbar(perm_dry["perm_mean"], perm_dry["feature"], xerr=perm_dry["perm_std"], fmt="o")
    ax4.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax4.set_title("DRY – permutation importance\n(ΔR² on full dataset)")
    ax4.set_xlabel("ΔR² when permuted")
    ax4.set_ylabel("")

    plt.tight_layout()
    return fig


def _extract_shap_values(shap_out):
    if isinstance(shap_out, list):
        if len(shap_out) == 1:
            return np.asarray(shap_out[0])
        return np.asarray(shap_out[-1])
    return np.asarray(shap_out)


def plot_fig6(
    target: str,
    model_wet: RandomForestRegressor,
    X_wet_df: pd.DataFrame,
    model_dry: RandomForestRegressor,
    X_dry_df: pd.DataFrame,
) -> plt.Figure:
    expl_wet = shap.TreeExplainer(model_wet)
    shap_wet = _extract_shap_values(expl_wet.shap_values(X_wet_df.values))

    expl_dry = shap.TreeExplainer(model_dry)
    shap_dry = _extract_shap_values(expl_dry.shap_values(X_dry_df.values))

    mean_abs_wet = np.mean(np.abs(shap_wet), axis=0)
    mean_abs_dry = np.mean(np.abs(shap_dry), axis=0)

    order_wet = np.argsort(mean_abs_wet)[::-1]
    order_dry = np.argsort(mean_abs_dry)[::-1]

    idx_naoi = FEATURES.index("NAOi")
    idx_pcp = FEATURES.index("pcp_sum")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axA, axB, axC, axD = axes.flatten()

    f_wet = [FEATURES[i] for i in order_wet]
    imp_wet = mean_abs_wet[order_wet]
    axA.barh(f_wet, imp_wet)
    axA.invert_yaxis()
    axA.set_xlabel(f"Mean |SHAP value| for {target} (‰)")
    axA.set_title("A) WET season – global SHAP importance", loc="left", fontsize=11, fontweight="bold")

    axB.axhline(0, color="0.7", linestyle="--", linewidth=1)
    axB.scatter(X_wet_df["NAOi"], shap_wet[:, idx_naoi], s=35, alpha=0.7)
    axB.set_xlabel("NAOi")
    axB.set_ylabel(f"SHAP value for {target} (‰)")
    axB.set_title("B) WET – SHAP dependence on NAOi", loc="left", fontsize=11, fontweight="bold")

    f_dry = [FEATURES[i] for i in order_dry]
    imp_dry = mean_abs_dry[order_dry]
    axC.barh(f_dry, imp_dry)
    axC.invert_yaxis()
    axC.set_xlabel(f"Mean |SHAP value| for {target} (‰)")
    axC.set_title("C) DRY season – global SHAP importance", loc="left", fontsize=11, fontweight="bold")

    axD.axhline(0, color="0.7", linestyle="--", linewidth=1)
    axD.scatter(X_dry_df["pcp_sum"], shap_dry[:, idx_pcp], s=35, alpha=0.7)
    axD.set_xlabel("Monthly precipitation (pcp_sum, mm)")
    axD.set_ylabel(f"SHAP value for {target} (‰)")
    axD.set_title("D) DRY – SHAP dependence on precipitation", loc="left", fontsize=11, fontweight="bold")

    fig.suptitle(f"SHAP-based interpretation of {target} drivers (WET vs DRY)", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def run_for_target(df: pd.DataFrame, target: str) -> None:
    df_wet = df[df["Season"] == "WET"].copy()
    df_dry = df[df["Season"] == "DRY"].copy()

    model_wet, fi_wet, perm_wet, metrics_wet, _, X_wet_df, _ = monte_carlo_rf_for_season(
        df_wet, "WET (ONDJFM)", target=target, features=FEATURES
    )
    model_dry, fi_dry, perm_dry, metrics_dry, _, X_dry_df, _ = monte_carlo_rf_for_season(
        df_dry, "DRY (AMJJAS)", target=target, features=FEATURES
    )

    models_dir = ROOT / "models" / target
    ensure_dir(models_dir)

    save_pickle(model_wet, models_dir / f"{target}_WET_RF.pkl")
    save_pickle(model_dry, models_dir / f"{target}_DRY_RF.pkl")

    write_metrics_txt(models_dir / f"{target}_WET_RF_metrics.txt", metrics_wet)
    write_metrics_txt(models_dir / f"{target}_DRY_RF_metrics.txt", metrics_dry)

    fig5 = plot_fig5(target, fi_wet, perm_wet, metrics_wet, fi_dry, perm_dry, metrics_dry)
    fig6 = plot_fig6(target, model_wet, X_wet_df, model_dry, X_dry_df)

    fig_dir = ROOT / "figures" / "models" / target
    ensure_dir(fig_dir)

    save_fig(
        fig5,
        fig_dir / f"RF_importance_WET_DRY_MonteCarlo_{target}.png",
        fig_dir / f"RF_importance_WET_DRY_MonteCarlo_{target}.svg",
        dpi=300,
        bbox_tight=False,
    )
    save_fig(
        fig6,
        fig_dir / f"SHAP_clean_panel_{target}.png",
        fig_dir / f"SHAP_clean_panel_{target}.svg",
        dpi=300,
        bbox_tight=True,
    )


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df["Date"] = parse_monthly_date(df["Date"])
    df["Season"] = df["Date"].dt.month.apply(season_from_month)

    for target in TARGETS:
        run_for_target(df, target)


if __name__ == "__main__":
    main()
