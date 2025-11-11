import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Success Sense – Startup Success Predictor", page_icon="🚀", layout="centered")

st.title("🚀 Success Sense")
st.caption("Predict your startup's success score (0–100).")

model_path = Path("../models/model.pkl")
cfg_path = Path("../models/feature_config.json")

if not model_path.exists():
    st.warning("Model not found. Please run **`python src/train.py`** from the project root to train the model.")
else:
    pipe = load(model_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with st.sidebar:
        st.header("Single Prediction")
        funding_amount = st.number_input("Funding Amount (USD)", min_value=0.0, value=2_000_000.0, step=10000.0, format="%.2f")
        market_size = st.number_input("Market Size (USD)", min_value=0.0, value=50_000_000.0, step=100000.0, format="%.2f")
        team_experience_years = st.number_input("Team Experience (years avg)", min_value=0.0, value=8.0, step=0.5)
        team_size = st.number_input("Team Size", min_value=1, value=15, step=1)
        avg_education_level = st.slider("Avg Education Level (0=HS, 1=UG, 2=PG, 3=PhD)", 0.0, 3.0, 2.0, 0.1)
        has_patent = st.selectbox("Has Patent", [0, 1])
        revenue_first_year = st.number_input("Revenue First Year (USD)", min_value=0.0, value=800_000.0, step=10000.0, format="%.2f")
        burn_rate = st.slider("Burn Rate (burn/revenue)", 0.1, 1.5, 0.6, 0.01)
        location_score = st.slider("Location Score (0-1)", 0.0, 1.0, 0.6, 0.01)
        competition_score = st.slider("Competition Score (0-1)", 0.0, 1.0, 0.5, 0.01)
        sector = st.selectbox("Sector", ["Fintech", "HealthTech", "EdTech", "E-Commerce", "AI", "SaaS"])

        if st.button("Predict"):
            row = pd.DataFrame([{
                "funding_amount": funding_amount,
                "market_size": market_size,
                "team_experience_years": team_experience_years,
                "team_size": team_size,
                "avg_education_level": avg_education_level,
                "has_patent": has_patent,
                "revenue_first_year": revenue_first_year,
                "burn_rate": burn_rate,
                "location_score": location_score,
                "competition_score": competition_score,
                "sector": sector
            }])
            pred = float(pipe.predict(row)[0])
            st.success(f"Predicted Success Score: **{pred:.1f} / 100**")

    st.divider()
    st.subheader("📦 Batch Prediction")
    st.write("Upload a CSV with the same columns used for training (without `success_score`).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        needed = cfg["numeric"] + cfg["categorical"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = pipe.predict(df[needed])
            out = df.copy()
            out["predicted_success_score"] = preds
            st.dataframe(out.head(20))
            st.download_button("Download Predictions CSV", out.to_csv(index=False), file_name="predictions.csv")