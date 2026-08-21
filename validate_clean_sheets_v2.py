"""
FPL Points Predictor — Clean Sheets Validation: Honest Comparison
========================================================================
Compares the CURRENT live clean sheet model (team-level Poisson,
possession/PPDA-rich, 25/26 only) against a SCOPED cross-season version
using only own/opp goals conceded/scored — an honestly more limited
feature set, since possession/PPDA don't exist in the 24/25 archive.

Run:
    python validate_clean_sheets_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75

SCOPED_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
                    "opp_season_goals_scored", "was_home_int"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_cs_model(train_df: pd.DataFrame, features: list, weight_col: str = None):
    feats = drop_zero_variance(train_df, [f for f in features if f in train_df.columns])
    if len(train_df) < 60 or not feats:
        return None, feats
    X = sm.add_constant(train_df[feats].fillna(0))
    fit_kwargs = {}
    if weight_col and weight_col in train_df.columns:
        fit_kwargs["freq_weights"] = train_df[weight_col]
    model = sm.GLM(train_df["goals_conceded"], X, family=sm.families.Poisson(), **fit_kwargs).fit()
    return model, feats


def predict_cs(df: pd.DataFrame, model, features: list) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=df.index)
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    X = X.reindex(columns=model.params.index, fill_value=0)
    return pd.Series(model.predict(X), index=df.index)


def main():
    path = PROCESSED_DIR / "clean_sheets_training_combined.csv"
    if not path.exists():
        print(f"{path} not found — run build_clean_sheets_training_data.py first.")
        return

    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train_current_only = df[(df["gameweek"] <= cutoff_gw) & (df["season"] == "2025-26")]
    train_combined = df[df["gameweek"] <= cutoff_gw]
    test = df[(df["gameweek"] > cutoff_gw) & (df["season"] == "2025-26")].copy()
    print(f"Train (current only): {len(train_current_only)} rows | "
          f"Train (combined, weighted): {len(train_combined)} rows | Test: {len(test)} rows\n")

    print("=" * 70)
    print("1. Current-season-only model (scoped features)")
    print("=" * 70)
    model_current, feats_current = fit_cs_model(train_current_only, SCOPED_FEATURES)
    print(f"  Features used: {feats_current}")
    test["pred_current"] = predict_cs(test, model_current, feats_current)

    print("\n" + "=" * 70)
    print("2. Cross-season weighted model (scoped features)")
    print("=" * 70)
    model_combined, feats_combined = fit_cs_model(train_combined, SCOPED_FEATURES, weight_col="sample_weight")
    print(f"  Features used: {feats_combined}")
    test["pred_combined"] = predict_cs(test, model_combined, feats_combined)

    evaluable = test.dropna(subset=["pred_current", "pred_combined", "goals_conceded"])
    print(f"\nEvaluable held-out rows: {len(evaluable)}\n")

    if len(evaluable) == 0:
        print("No evaluable rows — can't compare. Recommend leaving clean sheets as-is.")
        return

    mae_current = (evaluable["pred_current"] - evaluable["goals_conceded"]).abs().mean()
    mae_combined = (evaluable["pred_combined"] - evaluable["goals_conceded"]).abs().mean()

    print("=" * 70)
    print("HONEST COMPARISON — held-out MAE on goals conceded (lower is better)")
    print("=" * 70)
    print(f"  1. Current-season-only (scoped): {mae_current:.4f}")
    print(f"  2. Cross-season weighted (scoped): {mae_combined:.4f}")

    if mae_combined < mae_current:
        pct = (mae_current - mae_combined) / mae_current * 100
        print(f"\nCross-season is BETTER by {pct:.1f}%.")
    else:
        pct = (mae_combined - mae_current) / mae_current * 100
        print(f"\nCross-season is WORSE by {pct:.1f}%.")

    print(f"\nRemember: this compares a SCOPED feature set against itself across "
          f"seasons — NOT against the current LIVE model, which uses richer "
          f"possession/PPDA features unavailable here. Even if cross-season wins "
          f"this specific comparison, that doesn't mean it beats the live model — "
          f"it only tells us whether more data helps THIS scoped feature set. "
          f"A separate check against the live model's actual features is still "
          f"needed before recommending any live changes.")
    print(f"\nSame caution as before: a difference under roughly 5% could be noise.")


if __name__ == "__main__":
    main()
