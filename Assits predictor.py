"""
FPL Points Predictor — Predictor Strength Analysis: ASSISTS
================================================================
Same approach validated on goals, applied to assists. Starting directly
with a trimmed feature set (season_ + roll5_ windows only, not every
window) since we already learned that including 4 near-identical
windows of the same stat makes individual p-values unreliable due to
multicollinearity — no need to repeat that mistake here.

Assists are count data (mostly 0s, occasionally 1+), so Poisson
regression is again the right tool, same reasoning as for goals.

Run:
    python analyze_assists_predictors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")

CANDIDATE_FEATURES = [
    "season_xA", "roll5_xA",
    "season_key_passes", "roll5_key_passes",
    "season_xGChain", "roll5_xGChain",
    "season_xGBuildup", "roll5_xGBuildup",
    "season_Crs", "roll5_Crs",
    "season_shots", "roll5_shots",  # control for overall attacking involvement
    "season_xG", "roll5_xG",        # a good passer often also shoots — worth controlling for
    "opp_season_goals_conceded", "opp_roll5_goals_conceded",
    "opp_season_xG_against", "opp_roll5_xG_against",
    "was_home_int",
    "days_since_last_game",
    "roll5_minutes",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[(df["minutes"] > 0) & (df["roll5_xA"].notna())]
    return df


def correlation_ranking(df: pd.DataFrame) -> pd.Series:
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    corrs = df[available + ["assists"]].corr(numeric_only=True)["assists"].drop("assists")
    return corrs.sort_values(key=abs, ascending=False)


def poisson_regression(df: pd.DataFrame):
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y = df["assists"]

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
    y = df["assists"]

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    return pd.Series(rf.feature_importances_, index=available).sort_values(ascending=False)


def main():
    df = load_data()
    print(f"Rows used (played, with rolling history): {len(df)}")
    print(f"Assists distribution:\n{df['assists'].value_counts().sort_index()}\n")

    print("=" * 70)
    print("1. CORRELATION with assists")
    print("=" * 70)
    print(correlation_ranking(df).to_string())

    print("\n" + "=" * 70)
    print("2. POISSON REGRESSION coefficients (trimmed feature set)")
    print("=" * 70)
    model, result = poisson_regression(df)
    print(result.to_string())
    print(f"\nModel pseudo R-squared: {model.pseudo_rsquared():.4f}")

    print("\n" + "=" * 70)
    print("3. RANDOM FOREST feature importance")
    print("=" * 70)
    print(random_forest_importance(df).to_string())

    print("\nAs with goals: trust agreement across all three methods over any "
          "single one, and treat SHAP (separate script) as the tiebreaker "
          "for any features where Poisson and Random Forest disagree.")


if __name__ == "__main__":
    main()