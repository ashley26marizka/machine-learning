from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from utils import RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)

def synthesize(n=1200):
    sectors = ["Fintech", "HealthTech", "EdTech", "E-Commerce", "AI", "SaaS"]
    data = []
    for _ in range(n):
        sector = rng.choice(sectors)
        funding_amount = float(rng.lognormal(mean=14.5, sigma=0.9))  # ~ 2M to 100M
        market_size = float(rng.lognormal(mean=15.0, sigma=0.8))     # TAM proxy
        team_experience_years = float(rng.normal(12, 5))
        team_size = int(max(2, rng.normal(20, 10)))
        avg_education_level = float(np.clip(rng.normal(2.1, 0.5), 0, 3))  # 0=HS,1=UG,2=PG,3=PhD
        has_patent = int(rng.random() < 0.35)
        revenue_first_year = float(max(0, rng.normal(1_000_000, 500_000)))
        burn_rate = float(np.clip(rng.normal(0.65, 0.2), 0.1, 1.5))  # burn/revenue
        location_score = float(np.clip(rng.normal(0.6, 0.2), 0, 1))
        competition_score = float(np.clip(rng.normal(0.5, 0.25), 0, 1))

        # True underlying function (unknown to model)
        base = (
            0.30 * np.log1p(funding_amount) +
            0.25 * np.log1p(market_size) +
            0.15 * team_experience_years +
            0.08 * team_size +
            4.0 * avg_education_level +
            3.0 * has_patent +
            0.000004 * revenue_first_year +
            -12.0 * burn_rate +
            10.0 * location_score +
            -8.0 * competition_score
        )

        # Sector effects
        sector_bonus = {
            "Fintech": 4.0,
            "HealthTech": 5.0,
            "EdTech": 1.0,
            "E-Commerce": 2.0,
            "AI": 6.0,
            "SaaS": 3.0
        }[sector]

        noise = rng.normal(0, 6.0)
        success_score = float(np.clip(base + sector_bonus + noise, 0, 100))

        data.append({
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
            "sector": sector,
            "success_score": success_score
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    out = Path("data/startups.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = synthesize(n=1400)
    df.to_csv(out, index=False)
    # also write a small sample input without the target for batch prediction
    sample = df.drop(columns=["success_score"]).head(5)
    sample.to_csv("data/sample_input.csv", index=False)
    print(f"Wrote {out} with {len(df)} rows and data/sample_input.csv")