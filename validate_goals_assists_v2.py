"""
FPL Points Predictor — Goals/Assists Validation: Honest Comparison
========================================================================
Compares the CURRENT live goals/assists models (Poisson per position,
25/26 data only, rich Understat-derived features) against a
cross-season weighted version using whatever's genuinely shared between
both seasons.

Honest limitation, confirmed on real data: your current dataset has no
fpl_xG/fpl_xA columns, so the cross-season combined dataset only really
has position/was_home_int/goals_scored/assists in common — missing the
rich xG-based signal the CURRENT model actually relies on. This
validation will show whether even that limited combination helps, or
whether the current model (with its richer features) should just stay
as-is.

Run:
    python validate_goals_assists_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["DEF", "MID", "FWD"]
TRAIN_FRACTION = 0.75


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_by_position(train_df: pd.DataFrame, target: str, candidate_features: list, weight_col: str = None):
    models = {}
    for position in POSITIONS:
        pos_df = train_df[train_df["position"] == position]
        features = drop_zero_variance(pos_df, [f for f in candidate_features if f in pos_df.columns])
        if len(pos_df) < 60 or not features:
            print(f"    {position}: not enough data/features, skipping")
            continue
        X = sm.add_constant(pos_df[features].fillna(0))
        fit_kwargs = {}
        if weight_col and weight_col in pos_df.columns:
            fit_kwargs["freq_weights"] = pos_df[weight_col]
        try:
            models[position] = (sm.GLM(pos_df[target], X, family=sm.families.Poisson(), **fit_kwargs).fit(), features)
        except Exception as e:
            print(f"    {position}: fit failed ({e})")
    return models


def predict_by_position(df: pd.DataFrame, models: dict) -> pd.Series:
    preds = pd.Series(np.nan, index=df.index)
    for position, (model, features) in models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
        X = X.reindex(columns=model.params.index, fill_value=0)
        preds.loc[mask] = model.predict(X)
    return preds


def main():
    path = PROCESSED_DIR / "goals_assists_training_combined.csv"
    if not path.exists():
        print(f"{path} not found — run build_goals_assists_training_data.py first.")
        return

    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}")
    print(f"Columns available: {list(df.columns)}\n")

    candidate_features = [c for c in ["roll5_fpl_xG", "season_fpl_xG", "roll5_fpl_xA",
                                       "season_fpl_xA", "was_home_int"] if c in df.columns]
    print(f"Features available for the cross-season model: {candidate_features}\n")
    if not candidate_features:
        print("No usable shared features beyond position — cross-season enhancement "
              "isn't viable here with the current data. Recommend leaving goals/assists as-is.")
        return

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train_current_only = df[(df["gameweek"] <= cutoff_gw) & (df["season"] == "2025-26")]
    train_combined = df[df["gameweek"] <= cutoff_gw]
    test = df[(df["gameweek"] > cutoff_gw) & (df["season"] == "2025-26")].copy()
    print(f"Train (current only): {len(train_current_only)} rows | "
          f"Train (combined, weighted): {len(train_combined)} rows | Test: {len(test)} rows\n")

    for target in ["goals_scored", "assists"]:
        if target not in df.columns:
            print(f"'{target}' not in combined data — skipping.")
            continue

        print("=" * 70)
        print(f"TARGET: {target}")
        print("=" * 70)

        print("  Fitting current-season-only model...")
        models_current = fit_by_position(train_current_only, target, candidate_features)
        test[f"pred_{target}_current"] = predict_by_position(test, models_current)

        print("  Fitting cross-season weighted model...")
        models_combined = fit_by_position(train_combined, target, candidate_features, weight_col="sample_weight")
        test[f"pred_{target}_combined"] = predict_by_position(test, models_combined)

        evaluable = test.dropna(subset=[f"pred_{target}_current", f"pred_{target}_combined", target])
        if len(evaluable) == 0:
            print(f"  No evaluable rows for {target} — skipping comparison.\n")
            continue

        mae_current = (evaluable[f"pred_{target}_current"] - evaluable[target]).abs().mean()
        mae_combined = (evaluable[f"pred_{target}_combined"] - evaluable[target]).abs().mean()

        print(f"\n  Evaluable held-out rows: {len(evaluable)}")
        print(f"  Current-season-only MAE: {mae_current:.4f}")
        print(f"  Cross-season weighted MAE: {mae_combined:.4f}")
        if mae_combined < mae_current:
            pct = (mae_current - mae_combined) / mae_current * 100
            print(f"  Cross-season is BETTER by {pct:.1f}%\n")
        else:
            pct = (mae_combined - mae_current) / mae_current * 100
            print(f"  Cross-season is WORSE by {pct:.1f}%\n")

    print("Same caution as before: a difference under roughly 5% could be "
          "noise rather than a real improvement — only trust a clear margin.")


if __name__ == "__main__":
    main()
