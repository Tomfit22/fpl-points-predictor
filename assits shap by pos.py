"""
FPL Points Predictor — ASSISTS, by position, with sample-size-aware feature
selection and SHAP
=============================================================================
Same approach as analyze_goals_shap_by_position.py, applied to assists.
See that file's docstring for the full reasoning.

Run:
    python analyze_assists_shap_by_position.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["FWD", "MID", "DEF"]
EVENTS_PER_PREDICTOR = 15
MIN_FEATURES = 4

FULL_CANDIDATE_POOL = [
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


WINDOW_PREFIXES = ["season_", "roll10_", "roll5_", "roll3_"]


def get_feature_family(feature_name: str) -> str:
    """See analyze_goals_shap_by_position.py for full explanation."""
    name = feature_name
    opp_prefix = ""
    if name.startswith("opp_"):
        opp_prefix = "opp_"
        name = name[len("opp_"):]
    for prefix in WINDOW_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return opp_prefix + name


def select_features_for_position(pos_df: pd.DataFrame, n_events: int) -> list:
    n_feats = max(MIN_FEATURES, min(len(FULL_CANDIDATE_POOL), n_events // EVENTS_PER_PREDICTOR))
    available = [f for f in FULL_CANDIDATE_POOL if f in pos_df.columns]
    corrs = pos_df[available + ["assists"]].corr(numeric_only=True)["assists"].drop("assists").abs()

    corr_df = corrs.reset_index()
    corr_df.columns = ["feature", "abs_corr"]
    corr_df["family"] = corr_df["feature"].apply(get_feature_family)
    best_per_family = corr_df.sort_values("abs_corr", ascending=False).drop_duplicates(subset="family", keep="first")

    selected = best_per_family.sort_values("abs_corr", ascending=False).head(n_feats)["feature"].tolist()
    return selected, n_feats


def poisson_regression(df: pd.DataFrame, features: list):
    X = sm.add_constant(df[features].fillna(0))
    y = df["assists"]
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    coefs = model.params.drop("const")
    result = pd.DataFrame({
        "coefficient": coefs,
        "rate_ratio": np.exp(coefs),
        "p_value": model.pvalues.drop("const"),
    }).sort_values("p_value")
    return model, result


def shap_contribution(df: pd.DataFrame, features: list):
    X = df[features].fillna(0)
    y = df["assists"]

    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    contribution_pct = 100 * mean_abs_shap / mean_abs_shap.sum()

    ranking = pd.DataFrame({
        "feature": features,
        "contribution_pct": contribution_pct,
    }).sort_values("contribution_pct", ascending=False)
    return ranking


def analyze_position(df: pd.DataFrame, position: str):
    pos_df = df[df["position"] == position]
    n_events = (pos_df["assists"] > 0).sum()

    features, n_feats = select_features_for_position(pos_df, n_events)
    events_per_predictor = n_events / n_feats if n_feats else 0

    print("\n" + "#" * 70)
    print(f"# POSITION: {position}")
    print("#" * 70)
    print(f"Rows: {len(pos_df)} | Assist events: {n_events} | "
          f"Features selected: {n_feats} | Events/predictor: {events_per_predictor:.1f}")
    if events_per_predictor < 10:
        print("*** Still below the ideal 10-15 events/predictor even after trimming — "
              "treat results as directional, not statistically solid. ***")

    print(f"\nSelected features (top {n_feats}, one per family, by correlation with assists, this position only):")
    for f in features:
        print(f"  {f}  [family: {get_feature_family(f)}]")

    print(f"\n--- Poisson regression ---")
    model, result = poisson_regression(pos_df, features)
    print(result.to_string())
    print(f"Pseudo R-squared: {model.pseudo_rsquared():.4f}")

    print(f"\n--- SHAP contribution (fair credit-split across correlated predictors) ---")
    ranking = shap_contribution(pos_df, features)
    print(ranking.to_string(index=False))


def main():
    df = load_data()
    print(f"Total rows: {len(df)}")

    for position in POSITIONS:
        analyze_position(df, position)

    print("\n" + "=" * 70)
    print("Compare Poisson vs SHAP within each position — with far fewer, "
          "position-specific predictors, they should agree much more closely "
          "than the earlier full-candidate-list version did.")


if __name__ == "__main__":
    main()