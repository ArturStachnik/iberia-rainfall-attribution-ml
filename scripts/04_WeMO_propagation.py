from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError("Repository root not found (missing .project_root).")


ROOT = find_repo_root(Path(__file__))

MODEL_PATH = ROOT / "model" / "WeMO_PhysicallyConstrained_Multistation_XGB.pkl"

PREDICTORS_FILE = ROOT / "data" / "predictors" / "wemo_predictors_monthly_2010_2025_processed.csv"
TARGET_FILE = ROOT / "data" / "WeMO" / "WeMO_Bustins.csv"

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WEMO_DIR = ROOT / "data" / "WeMO"
WEMO_DIR.mkdir(parents=True, exist_ok=True)

FIG_PNG_PATH = FIG_DIR / "WeMO_Synthetic_1950_2025.png"
FIG_SVG_PATH = FIG_DIR / "WeMO_Synthetic_1950_2025.svg"
CSV_PATH = WEMO_DIR / "WeMO_Synthetic_1950_2025.csv"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


predictors = pd.read_csv(PREDICTORS_FILE)
predictors["Date"] = pd.to_datetime(predictors["Date"])

target = pd.read_csv(TARGET_FILE)
target["Date"] = pd.to_datetime(target["Date"])

df = predictors.merge(target, on="Date", how="left")

feature_cols = [
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
    "month_cos"
]

pred_mask = df[[
    "SLP_SanFernando",
    "SLP_Malo",
    "SLP_Teolo",
    "SLP_Venezia",
    "SLP_Cavallino"
]].notna().all(axis=1)

df["WeMO_ML"] = np.nan
df.loc[pred_mask, "WeMO_ML"] = model.predict(df.loc[pred_mask, feature_cols].values)


calib_mask = (
    (df["Date"] >= pd.Timestamp("2010-01-01")) &
    (df["Date"] <= pd.Timestamp("2020-12-01")) &
    df["WeMO"].notna() &
    df["WeMO_ML"].notna()
)

df_calib = df.loc[calib_mask].copy()

r2_full = r2_score(df_calib["WeMO"], df_calib["WeMO_ML"])
rmse_full = np.sqrt(mean_squared_error(df_calib["WeMO"], df_calib["WeMO_ML"]))


start_date = target["Date"].min()
end_date = pd.Timestamp("2025-12-01")

timeline = pd.date_range(start_date, end_date, freq="MS")
full_df = pd.DataFrame({"Date": timeline})

full_df = full_df.merge(target, on="Date", how="left")
full_df = full_df.merge(df[["Date", "WeMO_ML"]], on="Date", how="left")

full_df["WeMOi"] = np.where(
    full_df["Date"] <= pd.Timestamp("2020-12-01"),
    full_df["WeMO"],
    full_df["WeMO_ML"]
)

export_df = full_df[["Date", "WeMOi"]].copy()
export_df.to_csv(CSV_PATH, index=False)


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11
})

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

ax = axes[0]

ax.plot(df_calib["Date"], df_calib["WeMO"],
        color="black", lw=1.6,
        label="Observed WeMOi (Bustins, 2006)")

ax.plot(df_calib["Date"], df_calib["WeMO_ML"],
        color="#1f77b4", lw=1.6,
        label="Modelled WeMOi (this study)")

ax.set_title("A) Calibration interval (2010–2020): observed vs modelled WeMOi")
ax.set_ylabel("WeMO index")

metrics_text = (
    f"R² = {r2_full:.3f}\n"
    f"RMSE = {rmse_full:.3f}"
)

ax.text(0.02, 0.78, metrics_text, transform=ax.transAxes,
        fontsize=12, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.8))

ax.legend(loc="upper right")


ax2 = axes[1]

ax2.plot(full_df["Date"], full_df["WeMO"],
         color="black", lw=1,
         label="Observed WeMOi (1950–2020)")

ml_ext = full_df.copy()
ml_ext.loc[ml_ext["Date"] <= pd.Timestamp("2020-12-01"), "WeMO_ML"] = np.nan

ax2.plot(ml_ext["Date"], ml_ext["WeMO_ML"],
         color="#1f77b4", lw=1.4,
         label="ML extension (2021–2025)")

ax2.axvspan(pd.Timestamp("2021-01-01"),
            pd.Timestamp("2025-12-01"),
            color="#1f77b4", alpha=0.15)

ax2.set_title("B) Synthetic Western Mediterranean Oscillation index (WeMOi), 1950–2025")
ax2.set_ylabel("WeMO index")
ax2.set_xlabel("Year")

ax2.legend(loc="upper right")

plt.tight_layout()
fig.savefig(FIG_PNG_PATH, dpi=300, bbox_inches="tight")
fig.savefig(FIG_SVG_PATH, format="svg", bbox_inches="tight")
plt.close(fig)
