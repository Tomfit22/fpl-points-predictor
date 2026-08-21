"""
FPL Points Predictor — Clean Sheets Nonlinear Validation: Honest Comparison
========================================================================
Clean sheets already has opponent/possession/PPDA features validated
in earlier work (real Brier improvement: 0.1947->0.1917). The open
question left untested is whether a NONLINEAR model (Random Forest)
beats the current Poisson GLM on the SAME team-level features.

Compares via Brier score on the DERIVED clean-sheet probability
(P(goals_conceded=0), the actual quantity that matters for FPL scoring)
rather than raw MAE on goals conceded — same metric convention this
project already established for DC and clean sheets.

Run:
    python validate_clean_sheets_nonlinear.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75

CS_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found — run the main pipeline first.")
        return

    df = pd.read_csv(path)
    if "fixture_id" not in df.columns or "goals_conceded" not in df.columns:
        print("Missing fixture_id or goals_conceded — check the dataset.")
        return

    team_df = df.groupby(["team", "fixture_id", "gameweek"], as_index=False).first()
    print(f"Total team-match rows: {len(team_df)}\n")

    cutoff_gw = team_df["gameweek"].quantile(TRAIN_FRACTION)
    train = team_df[team_df["gameweek"] <= cutoff_gw]
    test = team_df[team_df["gameweek"] > cutoff_gw].copy()
    print(f"Train: {len(train)} rows (gw <= {cutoff_gw:.0f}) | Test: {len(test)} rows (gw > {cutoff_gw:.0f})\n")

    features = drop_zero_variance(train, [f for f in CS_FEATURES if f in train.columns])
    print(f"Features used: {features}\n")

    print("=" * 70)
    print("1. Current Poisson GLM (team-level)")
    print("=" * 70)
    X_train = sm.add_constant(train[features].fillna(0))
    poisson_model = sm.GLM(train["goals_conceded"], X_train, family=sm.families.Poisson()).fit()
    X_test = sm.add_constant(test[features].fillna(0), has_constant="add")
    X_test = X_test.reindex(columns=poisson_model.params.index, fill_value=0)
    test["pred_goals_poisson"] = poisson_model.predict(X_test)
    test["pred_cs_poisson"] = np.exp(-test["pred_goals_poisson"])  # P(goals_conceded=0)

    print("\n" + "=" * 70)
    print("2. Random Forest (same features)")
    print("=" * 70)
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(train[features].fillna(0), train["goals_conceded"])
    test["pred_goals_rf"] = rf.predict(test[features].fillna(0))
    test["pred_cs_rf"] = np.exp(-np.clip(test["pred_goals_rf"], 0, None))

    test["actual_cs"] = (test["goals_conceded"] == 0).astype(int)
    evaluable = test.dropna(subset=["pred_cs_poisson", "pred_cs_rf"])
    print(f"\nEvaluable held-out team-match rows: {len(evaluable)}\n")

    brier_poisson = ((evaluable["pred_cs_poisson"] - evaluable["actual_cs"]) ** 2).mean()
    brier_rf = ((evaluable["pred_cs_rf"] - evaluable["actual_cs"]) ** 2).mean()

    mae_poisson = (evaluable["pred_goals_poisson"] - evaluable["goals_conceded"]).abs().mean()
    mae_rf = (evaluable["pred_goals_rf"] - evaluable["goals_conceded"]).abs().mean()

    print("=" * 70)
    print("HONEST COMPARISON")
    print("=" * 70)
    print(f"Brier score on clean sheet probability (lower is better):")
    print(f"  1. Current Poisson: {brier_poisson:.4f}")
    print(f"  2. Random Forest:   {brier_rf:.4f}")
    if brier_rf < brier_poisson:
        pct = (brier_poisson - brier_rf) / brier_poisson * 100
        print(f"  Random Forest is BETTER by {pct:.1f}%")
    else:
        pct = (brier_rf - brier_poisson) / brier_poisson * 100
        print(f"  Random Forest is WORSE by {pct:.1f}%")

    print(f"\nSupplementary: MAE on goals conceded directly:")
    print(f"  1. Current Poisson: {mae_poisson:.4f}")
    print(f"  2. Random Forest:   {mae_rf:.4f}")

    print(f"\nSame caution as before: a difference under roughly 5% could be noise "
          f"rather than a real improvement — only trust a clear margin.")


if __name__ == "__main__":
    main()