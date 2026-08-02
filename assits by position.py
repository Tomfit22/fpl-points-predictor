"""
FPL Points Predictor — Predictor Strength Analysis: ASSISTS, BY POSITION
=============================================================================
Same position-segmented approach as analyze_goals_by_position.py, applied
to assists. See that file's docstring for the full reasoning on why this
matters and the sample-size caveat for less-frequent-scoring positions.

DEF is kept here (unlike a possible goals-only exclusion) since defenders
recording assists — often from crosses, set-piece deliveries, or long
progressive passes — is common enough to be worth modeling, even if less
frequent than for MID/FWD.

Run:
    python analyze_assists_by_position.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["FWD", "MID", "DEF"]
MIN_ASSIST_EVENTS_FOR_POISSON = 30

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


def correlation_ranking(df: pd.DataFrame, available: list) -> pd.Series:
    corrs = df[available + ["assists"]].corr(numeric_only=True)["assists"].drop("assists")
    return corrs.sort_values(key=abs, ascending=False)


def poisson_regression(df: pd.DataFrame, available: list):
    X = sm.add_constant(df[available].fillna(0))
    y = df["assists"]
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    coefs = model.params.drop("const")
    result = pd.DataFrame({
        "coefficient": coefs,
        "rate_ratio": np.exp(coefs),
        "p_value": model.pvalues.drop("const"),
    }).sort_values("p_value")
    return model, result


def random_forest_importance(df: pd.DataFrame, available: list) -> pd.Series:
    X = df[available].fillna(0)
    y = df["assists"]
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=available).sort_values(ascending=False)


def analyze_position(df: pd.DataFrame, position: str):
    pos_df = df[df["position"] == position]
    available = [f for f in CANDIDATE_FEATURES if f in pos_df.columns]
    n_assist_events = (pos_df["assists"] > 0).sum()

    print("\n" + "#" * 70)
    print(f"# POSITION: {position}")
    print("#" * 70)
    print(f"Rows: {len(pos_df)} | Assist-recording rows: {n_assist_events} "
          f"({n_assist_events / len(pos_df):.1%} of appearances)")

    if n_assist_events < MIN_ASSIST_EVENTS_FOR_POISSON:
        print(f"\n*** WARNING: only {n_assist_events} assist events for {position} — "
              f"below the {MIN_ASSIST_EVENTS_FOR_POISSON} threshold I'd want for any "
              f"real confidence in p-values below. Treat this position's Poisson "
              f"results as exploratory/directional only. ***")

    print(f"\n--- Correlation (top 10) ---")
    print(correlation_ranking(pos_df, available).head(10).to_string())

    print(f"\n--- Poisson regression (significant only, p<0.10) ---")
    model, result = poisson_regression(pos_df, available)
    sig = result[result["p_value"] < 0.10]
    if len(sig) > 0:
        print(sig.to_string())
    else:
        print("(none reached p<0.10 — with this sample size, that itself is informative)")
    print(f"Pseudo R-squared: {model.pseudo_rsquared():.4f}")

    print(f"\n--- Random Forest importance (top 10) ---")
    print(random_forest_importance(pos_df, available).head(10).to_string())


def main():
    df = load_data()
    print(f"Total rows across all outfield positions: {len(df)}")

    for position in POSITIONS:
        analyze_position(df, position)

    print("\n" + "=" * 70)
    print("Compare top predictors across positions — e.g. defenders' assists "
          "may lean more on crosses/set pieces than forwards', whose assists "
          "are more likely open-play through-balls or lay-offs.")


if __name__ == "__main__":
    main()