"""
FPL Points Predictor — BPS (Bonus Points System score), by position
=========================================================================
STAGE 1 of bonus point prediction. BPS is FPL's own underlying score,
computed from a documented formula built from goals/assists/defensive
actions/cards/clean sheets/minutes/etc — stats we already model
separately elsewhere in this pipeline. That makes BPS one of the more
learnable targets here: it should be well-predicted by a player's own
rolling versions of those same component stats.

BPS is NOT count data (it's a bounded-ish score that can occasionally
go negative for a poor match), so this uses ordinary least squares
(linear regression), not Poisson — different target shape, different
right tool, same reasoning we've applied throughout this project.

IMPORTANT — this predicts BPS, not bonus points (0-3) directly. Bonus
points depend on a player's BPS RANKED against the ~21 other players in
that SAME match — it's inherently relative, not an absolute threshold
like defensive contributions. Converting a BPS prediction into an
actual bonus prediction requires a full match-roster ranking step,
which needs a different data shape (all players in one fixture, not
one player's row in isolation) and is a separate follow-up script, not
solved here.

Run:
    python analyze_bps_by_position.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["GK", "DEF", "MID", "FWD"]
MIN_ROWS_FOR_REGRESSION = 60  # rough floor for a stable OLS fit given ~15-20 candidate features

FULL_CANDIDATE_POOL = [
    "season_bps", "roll5_bps",
    "season_goals", "roll5_goals",
    "season_assists", "roll5_assists",
    "season_xG", "roll5_xG",
    "season_xA", "roll5_xA",
    "season_def_contrib", "roll5_def_contrib",
    "season_TklW", "roll5_TklW",
    "season_Int", "roll5_Int",
    "season_clean_sheets", "roll5_clean_sheets",
    "season_CrdY", "roll5_CrdY",
    "roll5_minutes", "roll5_starts",
    "opp_season_shots_for", "opp_roll5_shots_for",
    "own_season_shots_against", "own_roll5_shots_against",
    "was_home_int",
    "days_since_last_game",
    # NEW: CBI directly feeds BPS as of the 2026/27 rule change (1 BPS per
    # 3 CBI, replacing the old 1-per-2 rate and removing the old tackle
    # penalty). Sourced from FPL's own API, zero legal risk. Same
    # staleness caveat already noted elsewhere in this project applies:
    # this model is trained on 2025/26 data under the OLD formula, so
    # will need retraining once real 2026/27 data accumulates — but
    # having the feature in place now means it's ready to pick up the
    # (now cleaner) CBI->BPS relationship as soon as that data exists.
    "season_CBI", "roll5_CBI",
]

WINDOW_PREFIXES = ["season_", "roll10_", "roll5_", "roll3_"]


def get_feature_family(feature_name: str) -> str:
    name = feature_name
    prefix = ""
    for p in ("opp_", "own_"):
        if name.startswith(p):
            prefix = p
            name = name[len(p):]
            break
    for w in WINDOW_PREFIXES:
        if name.startswith(w):
            name = name[len(w):]
            break
    return prefix + name


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    return df[(df["minutes"] > 0) & (df["roll5_bps"].notna())].copy()


def select_features_for_position(pos_df: pd.DataFrame, n_rows: int) -> list:
    n_feats = max(4, min(len(FULL_CANDIDATE_POOL), n_rows // 15))
    available = [f for f in FULL_CANDIDATE_POOL if f in pos_df.columns]

    # drop zero-variance columns first — a constant column (e.g. season_goals
    # for GK, who essentially never score) breaks OLS with a singular matrix
    # and produces meaningless NaN/zero coefficients rather than an error,
    # which is exactly what happened for GK before this fix.
    variances = pos_df[available].var(numeric_only=True)
    zero_var_cols = variances[variances.fillna(0) == 0].index.tolist()
    if zero_var_cols:
        print(f"  (dropping zero-variance columns for this position: {zero_var_cols})")
    available = [f for f in available if f not in zero_var_cols]

    corrs = pos_df[available + ["bps"]].corr(numeric_only=True)["bps"].drop("bps").abs()

    corr_df = corrs.reset_index()
    corr_df.columns = ["feature", "abs_corr"]
    corr_df["family"] = corr_df["feature"].apply(get_feature_family)
    best_per_family = corr_df.sort_values("abs_corr", ascending=False).drop_duplicates(subset="family", keep="first")

    selected = best_per_family.sort_values("abs_corr", ascending=False).head(n_feats)["feature"].tolist()
    return selected, n_feats


def ols_regression(df: pd.DataFrame, features: list):
    X = sm.add_constant(df[features].fillna(0))
    y = df["bps"]
    model = sm.OLS(y, X).fit()
    coefs = model.params.drop("const")
    result = pd.DataFrame({
        "coefficient": coefs,  # direct interpretation: +1 unit of predictor -> this many more/fewer BPS
        "p_value": model.pvalues.drop("const"),
    }).sort_values("p_value")
    return model, result


def shap_contribution(df: pd.DataFrame, features: list):
    X = df[features].fillna(0)
    y = df["bps"]

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    contribution_pct = 100 * mean_abs_shap / mean_abs_shap.sum()

    ranking = pd.DataFrame({
        "feature": features,
        "contribution_pct": contribution_pct,
    }).sort_values("contribution_pct", ascending=False)
    return ranking, rf


def analyze_position(df: pd.DataFrame, position: str):
    pos_df = df[df["position"] == position]

    print("\n" + "#" * 70)
    print(f"# POSITION: {position}")
    print("#" * 70)
    print(f"Rows: {len(pos_df)} | Mean BPS: {pos_df['bps'].mean():.2f} | "
          f"Std BPS: {pos_df['bps'].std():.2f} | Mean bonus: {pos_df['bonus'].mean():.2f}")

    if len(pos_df) < MIN_ROWS_FOR_REGRESSION:
        print(f"*** Only {len(pos_df)} rows — too few for a stable regression. Skipping. ***")
        return

    features, n_feats = select_features_for_position(pos_df, len(pos_df))
    print(f"Features selected: {len(features)} | Rows/predictor: {len(pos_df) / len(features):.1f}")

    print(f"\nSelected features (one per family):")
    for f in features:
        print(f"  {f}  [family: {get_feature_family(f)}]")

    print(f"\n--- OLS REGRESSION: predicting BPS ---")
    model, result = ols_regression(pos_df, features)
    print(result.to_string())
    print(f"R-squared: {model.rsquared:.4f} | Adjusted R-squared: {model.rsquared_adj:.4f}")

    print(f"\n--- SHAP contribution ---")
    ranking, rf = shap_contribution(pos_df, features)
    print(ranking.to_string(index=False))

    # quick real-world sanity check: how far off is a typical prediction?
    preds = rf.predict(pos_df[features].fillna(0))
    mae = np.abs(preds - pos_df["bps"]).mean()
    print(f"\nRandom Forest mean absolute error: {mae:.2f} BPS "
          f"(for context, mean BPS this position is {pos_df['bps'].mean():.2f})")


def main():
    df = load_data()
    print(f"Total rows: {len(df)}")

    for position in POSITIONS:
        analyze_position(df, position)

    print("\n" + "=" * 70)
    print("This predicts expected BPS per player. Converting to actual bonus "
          "points (0-3) requires ranking predicted BPS against the other "
          "players in the SAME real match — a separate follow-up step, "
          "since it needs full match rosters rather than single-player rows.")


if __name__ == "__main__":
    main()