# Success Sense — Startup Success Predictor

A resume-ready ML project that predicts a **Startup Success Score (0–100)** from features such as funding, market size, team metrics, and more.  
Includes a clean **Streamlit UI**, full training pipeline, and evaluation with **MAE** and **RMSE**.

## Tech Stack
**Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Streamlit**

## Project Structure
```
success-sense/
├── app/
│   └── streamlit_app.py          # Simple UI for single/batch prediction
├── data/
│   ├── startups.csv              # Synthetic training dataset
│   └── sample_input.csv          # Example for batch prediction
├── models/
│   ├── model.pkl                 # Trained model (created after training)
│   └── preprocessor.pkl          # Fitted preprocessor (created after training)
├── reports/
│   ├── metrics.json              # MAE/RMSE/R2
│   ├── feature_importance.png    # Model feature importance (permuted)
│   └── eda_correlation.png       # Correlation heatmap
├── src/
│   ├── make_dataset.py           # Synthetic dataset generator
│   ├── train.py                  # Train + evaluate + save artifacts
│   ├── evaluate.py               # Standalone evaluator
│   ├── predict.py                # CLI prediction for CSV
│   └── utils.py                  # Common helpers
├── requirements.txt
├── .gitignore
└── README.md
```

## Quickstart

```bash
# 1) Create & activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Regenerate dataset
python src/make_dataset.py

# 4) Train
python src/train.py

# 5) Run the UI
streamlit run app/streamlit_app.py
```

### CLI Prediction
```bash
python src/predict.py --input data/sample_input.csv --output predictions.csv
```

## Notes
- This repo uses a **RandomForestRegressor** with a **ColumnTransformer** for preprocessing.
- Metrics computed: **MAE** and **RMSE** (and R² for reference).
- The dataset is synthetic but realistic-looking and fully reproducible.