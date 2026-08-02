"""
FPL Points Predictor — SHAP Feature Contribution: ASSISTS
==============================================================
Same approach as analyze_goals_shap.py, applied to assists. See that
file's docstring for the full explanation of why SHAP is the right
tool for fairly splitting credit among correlated predictors.

Install:
    pip install shap
    (if you hit a numpy/numba version conflict, try: pip install "numpy<2.5")

Run:
    python analyze_assists_shap.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")

CANDIDATE_FEATURES = [
    "season_xA", "roll5_xA",
    "season_key_passes", "roll5_key_passes",
    "season_xGChain", "roll5_xGChain",
    "season_xGBuildup", "roll5_xGBuildup",
    "season_Crs", "roll5_Crs",
    "season_shots", "roll5_shots",
    "season_xG", "roll5_xG",
    "opp_season_goals_conceded", "opp_roll5_goals_conceded",
    "opp_season_xG_against", "opp_roll5_xG_against",
    "was_home_int",
    "days_since_last_game",
    "roll5_minutes",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    return df[(df["minutes"] > 0) & (df["roll5_xA"].notna())]


def main():
    df = load_data()
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y = df["assists"]

    print(f"Rows used: {len(df)}")
    print("Fitting Random Forest...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    print("Computing SHAP values (may take a minute)...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    contribution_pct = 100 * mean_abs_shap / mean_abs_shap.sum()

    ranking = pd.DataFrame({
        "feature": available,
        "mean_abs_shap": mean_abs_shap,
        "contribution_pct": contribution_pct,
    }).sort_values("contribution_pct", ascending=False)

    print("\n" + "=" * 70)
    print("OVERALL CONTRIBUTION — % share of total predictive weight, fairly split")
    print("=" * 70)
    print(ranking.to_string(index=False))
    print(f"\n(percentages sum to 100%: {ranking['contribution_pct'].sum():.1f})")

    print("\n" + "=" * 70)
    print("DIRECTION — does a HIGHER value push predicted assists up or down?")
    print("=" * 70)
    top_10 = ranking.head(10)["feature"].tolist()
    for feat in top_10:
        idx = available.index(feat)
        corr = np.corrcoef(X[feat], shap_values[:, idx])[0, 1]
        direction = "higher value -> MORE assists" if corr > 0 else "higher value -> FEWER assists"
        print(f"  {feat:<30} {direction} (corr={corr:.2f})")

    output_path = PROCESSED_DIR / "assists_shap_contributions.csv"
    ranking.to_csv(output_path, index=False)
    print(f"\nSaved full ranking -> {output_path}")


if __name__ == "__main__":
    main()