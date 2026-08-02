"""
FPL Points Predictor — Predictor Strength Analysis: GOALS (trimmed)
=======================================================================
Same as analyze_goals_predictors.py, but keeps only ONE window per stat
family (season_ for long-term quality, roll5_ for recent form) instead
of season/roll3/roll5/roll10 all at once. Cuts the multicollinearity
that was making individual Poisson p-values unreliable — with 4 near-
identical versions of xG in the same model, the model can't tell which
one deserves credit, which inflates standard errors for all of them.

Run:
    python analyze_goals_predictors_trimmed.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")

CANDIDATE_FEATURES = [
    "season_xG", "roll5_xG",
    "season_shots", "roll5_shots",
    "season_SoT", "roll5_SoT",
    "season_xGChain", "roll5_xGChain",
    "season_xGBuildup", "roll5_xGBuildup",
    "season_xA", "roll5_xA",
    "season_Fld", "roll5_Fld",
    "season_xG_per_shot", "roll5_xG_per_shot",
    "season_SoT_rate", "roll5_SoT_rate",
    "season_conversion_rate", "roll5_conversion_rate",
    "roll5_goals_per90", "roll5_xG_per90",
    "roll5_points", "roll5_bonus", "roll5_key_passes", "roll5_minutes",
    "opp_season_goals_conceded", "opp_roll5_goals_conceded",
    "opp_season_xG_against", "opp_roll5_xG_against",
    "was_home_int",
    "days_since_last_game",
    "is_primary_pen_taker", "is_backup_pen_taker", "season_PK_attempts",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[(df["minutes"] > 0) & (df["roll5_xG"].notna())]
    return df


def poisson_regression(df: pd.DataFrame):
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y = df["goals"]

    X = sm.add_constant(X)
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    coefs = model.params.drop("const")
    rate_ratios = np.exp(coefs)

    result = pd.DataFrame({
        "coefficient": coefs,
        "rate_ratio": rate_ratios,
        "p_value": model.pvalues.drop("const"),
    }).sort_values("p_value")

    return model, result


def random_forest_importance(df: pd.DataFrame) -> pd.Series:
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y = df["goals"]

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    return pd.Series(rf.feature_importances_, index=available).sort_values(ascending=False)


def main():
    df = load_data()
    print(f"Rows used: {len(df)}\n")

    print("=" * 70)
    print("POISSON REGRESSION — trimmed feature set (season_ + roll5_ only)")
    print("=" * 70)
    model, result = poisson_regression(df)
    print(result.to_string())
    print(f"\nModel pseudo R-squared: {model.pseudo_rsquared():.4f}")
    print("\nWith the redundant roll3/roll10 windows removed, these p-values are "
          "far more trustworthy individually — each predictor is no longer "
          "fighting near-duplicates of itself for credit.")

    print("\n" + "=" * 70)
    print("RANDOM FOREST feature importance — trimmed feature set")
    print("=" * 70)
    print(random_forest_importance(df).to_string())


if __name__ == "__main__":
    main()