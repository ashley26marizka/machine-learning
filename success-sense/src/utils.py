from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42

FEATURE_CONFIG = {
    "numeric": [
        "funding_amount",
        "market_size",
        "team_experience_years",
        "team_size",
        "avg_education_level",
        "has_patent",
        "revenue_first_year",
        "burn_rate",
        "location_score",
        "competition_score"
    ],
    "categorical": [
        "sector"
    ],
    "target": "success_score"
}

def load_feature_config():
    return FEATURE_CONFIG

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

def save_json(path: str | Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))