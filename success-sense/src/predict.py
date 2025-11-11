from __future__ import annotations
import argparse
import pandas as pd
from joblib import load
from utils import load_feature_config

def main():
    parser = argparse.ArgumentParser(description="Batch predict startup success scores")
    parser.add_argument("--input", required=True, help="CSV path with feature columns")
    parser.add_argument("--output", required=True, help="CSV path to write predictions")
    args = parser.parse_args()

    cfg = load_feature_config()
    pipe = load("models/model.pkl")

    df = pd.read_csv(args.input)
    X = df[cfg["numeric"] + cfg["categorical"]]
    preds = pipe.predict(X)
    out = df.copy()
    out["predicted_success_score"] = preds
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(out)} rows")

if __name__ == "__main__":
    main()