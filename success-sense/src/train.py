from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_feature_config, ensure_dirs, save_json, rmse, RANDOM_STATE

def main():
    ensure_dirs()
    cfg = load_feature_config()

    data_path = Path("data/startups.csv")
    if not data_path.exists():
        raise FileNotFoundError("data/startups.csv not found. Run: python src/make_dataset.py")

    df = pd.read_csv(data_path)

    # EDA correlation heatmap (numeric only)
    num_cols = cfg["numeric"] + [cfg["target"]]
    corr = df[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=False)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig("reports/eda_correlation.png")
    plt.close()

    X = df[cfg["numeric"] + cfg["categorical"]]
    y = df[cfg["target"]]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cfg["numeric"]),
            ("cat", categorical_transformer, cfg["categorical"]),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r = r2_score(y_test, preds)
    r_mse = rmse(y_test.values, preds)

    metrics = {"MAE": mae, "RMSE": r_mse, "R2": r}
    print("Metrics:", metrics)
    save_json("reports/metrics.json", metrics)

    # Permutation importance on a small subsample (for speed)
    X_test_small = X_test.sample(min(200, len(X_test)), random_state=RANDOM_STATE)
    y_test_small = y.loc[X_test_small.index]
    result = permutation_importance(pipe, X_test_small, y_test_small, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    importances = result.importances_mean

    # Get feature names after preprocessing
    ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(cfg["categorical"]))
    feature_names = cfg["numeric"] + cat_feature_names

    # Align importances length (ColumnTransformer order: numeric then categorical by our setup)
    imp_series = pd.Series(importances, index=feature_names[:len(importances)]) \
                .sort_values(ascending=False) \
                .head(20)


    plt.figure(figsize=(8,6))
    imp_series[::-1].plot(kind="barh")
    plt.title("Top Feature Importances (Permutation)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png")
    plt.close()

    # Save artifacts
    dump(pipe, "models/model.pkl")
    # (preprocessor is inside pipeline; saved for clarity if needed)
    # but if you want a separate preprocessor:
    dump(pipe.named_steps["preprocessor"], "models/preprocessor.pkl")

    # Save feature config for UI/CLI
    with open("models/feature_config.json", "w", encoding="utf-8") as f:
        json.dump({"numeric": cfg["numeric"], "categorical": cfg["categorical"]}, f, indent=2)

if __name__ == "__main__":
    main()