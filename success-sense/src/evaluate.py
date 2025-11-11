from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, r2_score
from utils import rmse, load_feature_config

def main():
    cfg = load_feature_config()
    model_path = Path("models/model.pkl")
    if not model_path.exists():
        raise FileNotFoundError("models/model.pkl not found. Run: python src/train.py")

    df = pd.read_csv("data/startups.csv")
    X = df[cfg["numeric"] + cfg["categorical"]]
    y = df[cfg["target"]]

    pipe = load(model_path)
    preds = pipe.predict(X)

    mae = mean_absolute_error(y, preds)
    r = r2_score(y, preds)
    r_mse = rmse(y.values, preds)
    metrics = {"MAE": mae, "RMSE": r_mse, "R2": r}
    print(metrics)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()