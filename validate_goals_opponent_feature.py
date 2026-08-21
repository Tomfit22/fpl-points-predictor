"""
FPL Points Predictor — Goals Model: Opponent Feature Bug Validation
========================================================================
Confirms and fixes a real bug found in GOALS_FEATURES: it uses
opp_season_shots_for (the opponent's OWN attacking shot volume) where
it should use opp_season_shots_against (how many shots the opponent
typically CONCEDES — the actual defensive-weakness signal). Confirmed
on real data these are meaningfully different, even inversely related
(-0.667 correlation) — a team that shoots a lot themselves tends to
concede fewer shots, not more, so the current feature could be pointing
in roughly the wrong direction for "is this an easy defense to score
against".

ASSISTS_FEATURES already gets this right (uses opp_season_goals_conceded
correctly) — this brings GOALS_FEATURES in line with the same, already
sensible pattern.

Compares three versions on genuine held-out (time-based) data:
  1. CURRENT   — opp_season_shots_for (the likely-buggy original)
  2. FIXED_SHOTS — opp_season_shots_against (parallel to the player's
                    own season_shots/roll5_shots features)
  3. FIXED_GOALS — opp_season_goals_conceded (matches ASSISTS_FEATURES'
                    existing convention exactly)

Run:
    python validate_goals_opponent_feature.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["DEF", "MID", "FWD"]
TRAIN_FRACTION = 0.75

VERSIONS = {
    "1_CURRENT": ["season_xG", "roll5_xG", "season_shots", "roll5_shots",
                  "opp_season_shots_for", "was_home_int", "is_primary_pen_taker"],
    "2_FIXED_SHOTS": ["season_xG", "roll5_xG", "season_shots", "roll5_shots",
                      "opp_season_shots_against", "was_home_int", "is_primary_pen_taker"],
    "3_FIXED_GOALS": ["season_xG", "roll5_xG", "season_shots", "roll5_shots",
                      "opp_season_goals_conceded", "was_home_int", "is_primary_pen_taker"],
}


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_and_evaluate(version_name, candidate_features, train, test):
    total_abs_err = 0.0
    total_rows = 0
    per_position = {}

    for position in POSITIONS:
        pos_train = train[train["position"] == position]
        pos_test = test[test["position"] == position]
        features = drop_zero_variance(pos_train, [f for f in candidate_features if f in pos_train.columns])
        if len(pos_train) < 60 or not features:
            continue

        X_train = sm.add_constant(pos_train[features].fillna(0))
        model = sm.GLM(pos_train["goals"], X_train, family=sm.families.Poisson()).fit()

        X_test = sm.add_constant(pos_test[features].fillna(0), has_constant="add")
        X_test = X_test.reindex(columns=model.params.index, fill_value=0)
        preds = model.predict(X_test)

        abs_err = (preds - pos_test["goals"]).abs().sum()
        total_abs_err += abs_err
        total_rows += len(pos_test)

        opp_feat = [f for f in features if f.startswith("opp_")]
        if opp_feat:
            coef = model.params.get(opp_feat[0], float("nan"))
            pval = model.pvalues.get(opp_feat[0], float("nan"))
            per_position[position] = (opp_feat[0], coef, pval)

    mae = total_abs_err / total_rows if total_rows > 0 else float("nan")
    return mae, per_position


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)

    goals_col = "goals" if "goals" in df.columns else ("goals_scored" if "goals_scored" in df.columns else None)
    if goals_col is None:
        print("No 'goals' or 'goals_scored' column found.")
        return
    if goals_col != "goals":
        df = df.rename(columns={goals_col: "goals"})

    if "gameweek" not in df.columns:
        print("No gameweek column - cannot do a time-based split.")
        return

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw]

    print(f"Train: {len(train)} rows (gw <= {cutoff_gw:.0f}) | Test: {len(test)} rows (gw > {cutoff_gw:.0f})\n")

    results = {}
    for version_name, features in VERSIONS.items():
        print(f"--- {version_name} ---")
        mae, per_position = fit_and_evaluate(version_name, features, train, test)
        print(f"  Held-out MAE (goals): {mae:.4f}")
        for position, (feat, coef, pval) in per_position.items():
            sig = "significant" if pval < 0.05 else "not significant"
            print(f"    {position}: {feat} coef={coef:+.4f}, p={pval:.4f} ({sig})")
        results[version_name] = mae
        print()

    print("=" * 70)
    print("FINAL VERDICT - ranked by held-out MAE (lower is better)")
    print("=" * 70)
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        best = "  <-- BEST" if mae == min(results.values()) else ""
        print(f"  {name}: MAE={mae:.4f}{best}")


if __name__ == "__main__":
    main()
