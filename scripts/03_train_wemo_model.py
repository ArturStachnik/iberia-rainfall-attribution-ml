from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError("Repository root not found (missing .project_root).")


ROOT = find_repo_root(Path(__file__))


PREDICTORS_FILE = ROOT / "data" / "predictors" / "wemo_predictors_monthly_2010_2025_processed.csv"
TARGET_FILE = ROOT / "data" / "WeMO" / "WeMO_Bustins.csv"

MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "WeMO_PhysicallyConstrained_Multistation_XGB"
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.pkl"
METRICS_PATH = MODEL_DIR / f"{MODEL_NAME}_metrics.txt"


predictors = pd.read_csv(PREDICTORS_FILE)
predictors["Date"] = pd.to_datetime(predictors["Date"])

target = pd.read_csv(TARGET_FILE)
target["Date"] = pd.to_datetime(target["Date"])

df = predictors.merge(target, on="Date", how="inner")
df = df[df["Date"] <= pd.Timestamp("2020-12-01")].copy()
df = df.dropna()

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

X = df[feature_cols].values
y = df["WeMO"].values


base_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

param_dist = {
    "n_estimators":     [200, 400, 600, 800, 1000],
    "max_depth":        [2, 3, 4, 5],
    "learning_rate":    [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":            [0.0, 0.1, 0.2],
    "reg_lambda":       [0.5, 1.0, 1.5, 2.0],
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=60,
    scoring="neg_mean_squared_error",
    cv=inner_cv,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X, y)

best_params = random_search.best_params_

outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []
r2_scores = []

for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        **best_params
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))


rmse_mean = np.mean(rmse_scores)
rmse_std = np.std(rmse_scores)

mae_mean = np.mean(mae_scores)
mae_std = np.std(mae_scores)

r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)


final_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    **best_params
)

final_model.fit(X, y)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(final_model, f)


with open(METRICS_PATH, "w") as f:
    f.write(f"{MODEL_NAME}\n\n")
    f.write("Optimal hyperparameters:\n")
    for k, v in best_params.items():
        f.write(f"{k} = {v}\n")
    f.write("\n")
    f.write(f"Number of samples = {len(y)}\n")
    f.write(f"Number of predictors = {X.shape[1]}\n\n")
    f.write("10-fold Cross-Validation Results\n")
    f.write(f"RMSE_mean = {rmse_mean:.4f}\n")
    f.write(f"RMSE_std  = {rmse_std:.4f}\n\n")
    f.write(f"MAE_mean  = {mae_mean:.4f}\n")
    f.write(f"MAE_std   = {mae_std:.4f}\n\n")
    f.write(f"R2_mean   = {r2_mean:.4f}\n")
    f.write(f"R2_std    = {r2_std:.4f}\n")
