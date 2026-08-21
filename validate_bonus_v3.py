"""
FPL Points Predictor — Bonus Validation: 3-Way Honest Comparison
========================================================================
Compares THREE approaches to bonus prediction on the SAME genuinely
held-out (time-based) data:

  1. Linear (player only)   — the CURRENT live model (bonus v2),
                               already validated at 13.9% better than
                               the flat position-average baseline
  2. Linear (+opponent)     — same Poisson approach, adding
                               opp_season_shots_for / own_season_
                               possession / own_season_ppda — features
                               already proven useful for DC/clean
                               sheets elsewhere in this project
  3. Random Forest (+opponent) — nonlinear, same feature set as #2

Note: the opponent/possession/PPDA features are Understat/FBref-only
and don't exist in the 24/25 archive — they'll be sparse for the
24/25 rows (honest limitation, not a bug; see
build_bonus_training_data_v2.py).

Run:
    python validate_bonus_v3.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["GK", "DEF", "MID", "FWD"]
TRAIN_FRACTION = 0.75

PLAYER_ONLY_FEATURES = ["roll5_bps", "season_bps", "roll5_CrdY", "roll5_minutes", "was_home_int"]
WITH_OPPONENT_FEATURES = PLAYER_ONLY_FEATURES + ["opp_season_shots_for", "own_season_possession", "own_season_ppda"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_linear_by_position(train_df: pd.DataFrame, features: list):
    models = {}
    for position in POSITIONS:
        pos_df = train_df[train_df["position"] == position]
        feats = drop_zero_variance(pos_df, [f for f in features if f in pos_df.columns])
        if len(pos_df) < 60 or not feats:
            print(f"    {position}: not enough data/features, skipping")
            continue
        X = sm.add_constant(pos_df[feats].fillna(0))
        weights = pos_df["sample_weight"] if "sample_weight" in pos_df.columns else None
        fit_kwargs = {"freq_weights": weights} if weights is not None else {}
        try:
            models[position] = (sm.GLM(pos_df["bonus"], X, family=sm.families.Poisson(), **fit_kwargs).fit(), feats)
        except Exception as e:
            print(f"    {position}: fit failed ({e})")
    return models


def fit_rf_by_position(train_df: pd.DataFrame, features: list):
    models = {}
    for position in POSITIONS:
        pos_df = train_df[train_df["position"] == position]
        feats = drop_zero_variance(pos_df, [f for f in features if f in pos_df.columns])
        if len(pos_df) < 60 or not feats:
            print(f"    {position}: not enough data/features, skipping")
            continue
        X = pos_df[feats].fillna(0)
        y = pos_df["bonus"]
        weights = pos_df["sample_weight"] if "sample_weight" in pos_df.columns else None
        rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y, sample_weight=weights)
        models[position] = (rf, feats)
    return models


def predict_linear(df: pd.DataFrame, models: dict) -> pd.Series:
    preds = pd.Series(np.nan, index=df.index)
    for position, (model, features) in models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
        X = X.reindex(columns=model.params.index, fill_value=0)
        preds.loc[mask] = model.predict(X)
    return preds


def predict_rf(df: pd.DataFrame, models: dict) -> pd.Series:
    preds = pd.Series(np.nan, index=df.index)
    for position, (model, features) in models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = df.loc[mask, features].fillna(0)
        preds.loc[mask] = model.predict(X)
    return preds


def main():
    path = PROCESSED_DIR / "bonus_training_combined_v2.csv"
    if not path.exists():
        print(f"{path} not found — run build_bonus_training_data_v2.py first.")
        return

    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}\n")

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw].copy()
    print(f"Train: {len(train)} rows (gw <= {cutoff_gw:.0f}) | Test: {len(test)} rows (gw > {cutoff_gw:.0f})\n")

    print("=" * 70)
    print("1. Linear, player features only (CURRENT live model, bonus v2)")
    print("=" * 70)
    models_linear_player = fit_linear_by_position(train, PLAYER_ONLY_FEATURES)
    test["pred_linear_player"] = predict_linear(test, models_linear_player)

    print("\n" + "=" * 70)
    print("2. Linear, WITH opponent/possession/PPDA features")
    print("=" * 70)
    models_linear_opp = fit_linear_by_position(train, WITH_OPPONENT_FEATURES)
    test["pred_linear_opp"] = predict_linear(test, models_linear_opp)

    print("\n" + "=" * 70)
    print("3. Random Forest, WITH opponent/possession/PPDA features")
    print("=" * 70)
    models_rf = fit_rf_by_position(train, WITH_OPPONENT_FEATURES)
    test["pred_rf"] = predict_rf(test, models_rf)

    evaluable = test.dropna(subset=["pred_linear_player", "pred_linear_opp", "pred_rf", "bonus"])
    print(f"\nEvaluable held-out rows (all 3 approaches have predictions): {len(evaluable)}\n")

    results = {}
    for name, col in [("1. Linear (player only)", "pred_linear_player"),
                       ("2. Linear (+opponent)", "pred_linear_opp"),
                       ("3. Random Forest (+opponent)", "pred_rf")]:
        mae = (evaluable[col] - evaluable["bonus"]).abs().mean()
        results[name] = mae

    print("=" * 70)
    print("HONEST COMPARISON — held-out MAE (lower is better)")
    print("=" * 70)
    best_name = min(results, key=results.get)
    for name, mae in results.items():
        marker = "  <-- BEST" if name == best_name else ""
        print(f"  {name}: {mae:.4f}{marker}")

    reference_mae = results["1. Linear (player only)"]
    print(f"\nRelative to current live model ({reference_mae:.4f}):")
    for name, mae in results.items():
        if name == "1. Linear (player only)":
            continue
        if mae < reference_mae:
            pct = (reference_mae - mae) / reference_mae * 100
            print(f"  {name}: {pct:.1f}% better")
        else:
            pct = (mae - reference_mae) / reference_mae * 100
            print(f"  {name}: {pct:.1f}% WORSE")

    print(f"\nNote: the opponent/possession/PPDA features are only present "
          f"for 25/26 rows (not in the 24/25 archive) — if the improvement "
          f"here is small, part of that may be because roughly a third of "
          f"the training data (24/25) can't use these features at all, "
          f"diluting their impact rather than the features being weak.")
    print(f"\nSame caution as before: a difference under roughly 5% could be "
          f"noise rather than a real improvement — only trust a clear margin.")


if __name__ == "__main__":
    main()