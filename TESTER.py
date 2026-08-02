"""
FPL Points Predictor — CLEAN SHEETS (final version)
==========================================================
Supersedes the earlier build_clean_sheets_model.py. Two fixes from what
we learned in validate_clean_sheets_calibration.py:

  1. BASELINE LEAKAGE BUG, now fixed: the original naive baseline used
     test["kept_clean_sheet"].mean() — the TEST set's own true average —
     as the prediction value. That's leakage: it secretly uses
     information a real forecaster wouldn't have in advance. Fixed to
     use the TRAINING set's rate instead, applied to test.

  2. PRIMARY METHOD CHANGED to Poisson regression on goals conceded,
     deriving clean-sheet probability as P(X=0) = e^-lambda from the
     fitted rate. This beat both the direct logistic classifier and
     calibrated versions of it in validation (Brier 0.195 vs 0.203
     best-classifier vs 0.221 team's-own-rate baseline).

Run:
    python build_clean_sheets_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75

CANDIDATE_POOL = [
    "own_season_goals_conceded", "own_roll5_goals_conceded",
    "own_season_xG_against", "own_roll5_xG_against",
    "own_season_shots_against", "own_roll5_shots_against",
    "opp_season_goals_scored", "opp_roll5_goals_scored",
    "opp_season_xG_for", "opp_roll5_xG_for",
    "opp_season_shots_for", "opp_roll5_shots_for",
    "was_home_int",
]


def build_team_match_table() -> pd.DataFrame:
    """goals_conceded via max() per team-match — NOT an arbitrary
    drop_duplicates row, which could land on a substitute's truncated
    on-pitch view and silently misrepresent the real match result."""
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    own_opp_cols = [c for c in CANDIDATE_POOL if c in df.columns and c != "was_home_int"]

    goals_conceded = df.groupby(["team", "fixture_id"])["goals_conceded"].max().reset_index()
    other_cols = df.groupby(["team", "fixture_id"])[
        ["gameweek", "opponent_team", "match_date", "was_home_int"] + own_opp_cols
    ].first().reset_index()

    team_df = goals_conceded.merge(other_cols, on=["team", "fixture_id"])
    team_df["kept_clean_sheet"] = (team_df["goals_conceded"] == 0).astype(int)
    return team_df.sort_values("gameweek")


def select_features(df: pd.DataFrame) -> list:
    """Both season_/roll5_ windows, pruning only genuine near-duplicates
    (correlation > 0.98) — validated as safe and slightly better than
    forcing one-per-family with this small a candidate pool."""
    available = [f for f in CANDIDATE_POOL if f in df.columns]
    selected = []
    for f in available:
        too_similar = any(abs(df[f].corr(df[g])) > 0.98 for g in selected)
        if not too_similar:
            selected.append(f)
    return selected


def brier(y_true, y_prob) -> float:
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def poisson_model(train: pd.DataFrame, features: list):
    X_train = sm.add_constant(train[features].fillna(0))
    y_train = train["goals_conceded"]
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

    coefs = model.params.drop("const")
    result = pd.DataFrame({
        "coefficient": coefs,
        "rate_ratio": np.exp(coefs),  # >1 = increases expected goals conceded (BAD for clean sheet), <1 = decreases
        "p_value": model.pvalues.drop("const"),
    }).sort_values("p_value")
    return model, result


def shap_contribution(df: pd.DataFrame, features: list):
    """SHAP on an RF predicting goals_conceded directly — explains the same
    underlying quantity the Poisson model predicts, for feature-importance
    context alongside the Poisson coefficients."""
    X = df[features].fillna(0)
    y = df["goals_conceded"]

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


def evaluate_out_of_sample(df: pd.DataFrame, features: list):
    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw]

    model, _ = poisson_model(train, features)
    X_test = sm.add_constant(test[features].fillna(0), has_constant="add")
    predicted_rate = model.predict(X_test)
    predicted_p_clean_sheet = np.exp(-predicted_rate)

    model_brier = brier(test["kept_clean_sheet"], predicted_p_clean_sheet)

    # FIXED baseline: training set's rate, NOT test set's own true rate
    train_rate = train["kept_clean_sheet"].mean()
    naive_brier = brier(test["kept_clean_sheet"], train_rate)

    print(f"\nTrain: {len(train)} team-matches (gw <= {cutoff_gw:.0f}) | "
          f"Test: {len(test)} team-matches (gw > {cutoff_gw:.0f})")
    print(f"Naive baseline (training-set clean sheet rate, {train_rate:.1%}, applied to every test row): "
          f"Brier = {naive_brier:.4f}")
    print(f"Poisson-derived model: Brier = {model_brier:.4f}")
    if model_brier >= naive_brier:
        print("*** WARNING: model does NOT beat the (correctly computed, non-leaky) naive baseline. ***")
    else:
        print(f"Model beats the naive baseline by {naive_brier - model_brier:.4f} Brier points.")

    accuracy = ((predicted_p_clean_sheet > 0.5).astype(int) == test["kept_clean_sheet"]).mean()
    naive_accuracy = ((train_rate > 0.5) == test["kept_clean_sheet"]).mean()
    print(f"\n(For reference — accuracy: model = {accuracy:.1%}, naive = {naive_accuracy:.1%}. "
          f"Brier score is the more meaningful metric here since the real use case is a "
          f"probability, not a forced yes/no guess.)")


def main():
    df = build_team_match_table()
    df = df[df["own_roll5_goals_conceded"].notna()]
    print(f"Total team-match rows: {len(df)}")
    print(f"Overall clean sheet rate: {df['kept_clean_sheet'].mean():.1%}")

    features = select_features(df)
    print(f"\nFeatures used: {len(features)}")
    for f in features:
        print(f"  {f}")

    print(f"\n--- POISSON REGRESSION: predicting goals conceded (primary method) ---")
    full_model, result = poisson_model(df, features)
    print(result.to_string())
    print(f"Pseudo R-squared: {full_model.pseudo_rsquared():.4f}")

    print(f"\n--- SHAP contribution (on goals conceded) ---")
    ranking = shap_contribution(df, features)
    print(ranking.to_string(index=False))

    print(f"\n--- OUT-OF-SAMPLE VALIDATION (fixed, non-leaky baseline) ---")
    evaluate_out_of_sample(df, features)

    print("\n" + "=" * 70)
    print("P(clean sheet) = exp(-predicted goals conceded). Converting to an "
          "individual player's points still requires knowing whether THAT "
          "player will play 60+ minutes — a separate prediction. Multiply: "
          "P(clean sheet) x P(plays 60+) x position value "
          "(4 for GK/DEF, 1 for MID, 0 for FWD) for an expected-points estimate.")


if __name__ == "__main__":
    main()