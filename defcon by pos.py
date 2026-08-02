
"""
FPL Points Predictor — DEFENSIVE CONTRIBUTIONS, by position
=================================================================
Different structure from goals/assists: the FPL points payoff isn't
linear in defensive actions — it's a STEP FUNCTION. A defender gets 2
points at 10+ combined tackles/interceptions/blocks/clearances that
match, a midfielder/forward needs 12+. Nothing below the threshold,
flat 2 points at/above it, regardless of how far past it they go.

That means the binary question — "did they cross the threshold?" — is
what actually earns points, and gets the PRIMARY treatment here via
logistic regression + SHAP. The raw count is modeled too (Poisson,
same approach as goals/assists) as a secondary/comparison view, since
it's still useful context even though it's not what directly pays out.

GK is excluded — defensive contribution points don't apply to
goalkeepers.

Run:
    python analyze_defcontrib_by_position.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["DEF", "MID", "FWD"]
EVENTS_PER_PREDICTOR = 15
MIN_FEATURES = 4

DC_THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12}  # official FPL thresholds by position

FULL_CANDIDATE_POOL = [
    "season_def_contrib", "roll5_def_contrib",
    "season_TklW", "roll5_TklW",
    "season_Int", "roll5_Int",
    "season_Fld", "roll5_Fld",
    "season_Fls", "roll5_Fls",
    "roll5_minutes", "roll5_starts",
    "opp_season_shots_for", "opp_roll5_shots_for",
    "own_season_shots_against", "own_roll5_shots_against",
    "was_home_int",
    "days_since_last_game",
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
    df = df[(df["minutes"] > 0) & (df["roll5_def_contrib"].notna()) & (df["position"] != "GK")].copy()
    df["hit_dc_threshold"] = df.apply(
        lambda row: int(row["defensive_contribution"] >= DC_THRESHOLDS[row["position"]]), axis=1
    )
    return df


def select_features_for_position(pos_df: pd.DataFrame, n_events: int, target_col: str) -> list:
    n_feats = max(MIN_FEATURES, min(len(FULL_CANDIDATE_POOL), n_events // EVENTS_PER_PREDICTOR))
    available = [f for f in FULL_CANDIDATE_POOL if f in pos_df.columns]
    corrs = pos_df[available + [target_col]].corr(numeric_only=True)[target_col].drop(target_col).abs()

    corr_df = corrs.reset_index()
    corr_df.columns = ["feature", "abs_corr"]
    corr_df["family"] = corr_df["feature"].apply(get_feature_family)
    best_per_family = corr_df.sort_values("abs_corr", ascending=False).drop_duplicates(subset="family", keep="first")

    selected = best_per_family.sort_values("abs_corr", ascending=False).head(n_feats)["feature"].tolist()
    return selected, n_feats


def logistic_regression(df: pd.DataFrame, features: list):
    X = sm.add_constant(df[features].fillna(0))
    y = df["hit_dc_threshold"]
    model = sm.Logit(y, X).fit(disp=0)
    coefs = model.params.drop("const")
    result = pd.DataFrame({
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),  # >1 = increases odds of hitting threshold, <1 = decreases
        "p_value": model.pvalues.drop("const"),
    }).sort_values("p_value")
    return model, result


def poisson_regression(df: pd.DataFrame, features: list):
    X = sm.add_constant(df[features].fillna(0))
    y = df["defensive_contribution"]
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    coefs = model.params.drop("const")
    result = pd.DataFrame({
        "coefficient": coefs,
        "rate_ratio": np.exp(coefs),
        "p_value": model.pvalues.drop("const"),
    }).sort_values("p_value")
    return model, result


def shap_contribution_binary(df: pd.DataFrame, features: list):
    X = df[features].fillna(0)
    y = df["hit_dc_threshold"]

    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    # handle both shap API conventions (list of per-class arrays, or 3D array)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    contribution_pct = 100 * mean_abs_shap / mean_abs_shap.sum()

    ranking = pd.DataFrame({
        "feature": features,
        "contribution_pct": contribution_pct,
    }).sort_values("contribution_pct", ascending=False)
    return ranking


def analyze_position(df: pd.DataFrame, position: str):
    pos_df = df[df["position"] == position]
    n_hit = pos_df["hit_dc_threshold"].sum()
    threshold = DC_THRESHOLDS[position]

    print("\n" + "#" * 70)
    print(f"# POSITION: {position} (threshold: {threshold}+ defensive actions = 2 points)")
    print("#" * 70)
    print(f"Rows: {len(pos_df)} | Hit threshold: {n_hit} ({n_hit / len(pos_df):.1%} of appearances) | "
          f"Mean defensive_contribution: {pos_df['defensive_contribution'].mean():.2f}")

    features, n_feats = select_features_for_position(pos_df, n_hit, "hit_dc_threshold")
    events_per_predictor = n_hit / n_feats if n_feats else 0
    print(f"Features selected: {n_feats} | Events/predictor: {events_per_predictor:.1f}")
    if events_per_predictor < 10:
        print("*** Below the ideal 10-15 events/predictor — treat as directional, not solid. ***")

    print(f"\nSelected features (one per family):")
    for f in features:
        print(f"  {f}  [family: {get_feature_family(f)}]")

    MIN_ABSOLUTE_EVENTS = 20  # below this, logistic regression breaks down entirely (perfect separation), not just weaker
    if n_hit < MIN_ABSOLUTE_EVENTS:
        print(f"\n*** Only {n_hit} threshold-hit events for {position} — too few to fit a logistic "
              f"regression meaningfully (risk of perfect separation / nonsense output). Skipping "
              f"logistic regression and SHAP for this position. The Poisson regression on the raw "
              f"count below is still shown, since it doesn't have this same failure mode, but treat "
              f"even that as exploratory only for this position. ***")
        _, poisson_result = poisson_regression(pos_df, features)
        print(f"\n--- POISSON REGRESSION: raw defensive_contribution count (only reliable view here) ---")
        print(poisson_result.to_string())
        return

    print(f"\n--- LOGISTIC REGRESSION: P(hit {threshold}+ threshold) ---")
    logit_model, logit_result = logistic_regression(pos_df, features)
    print(logit_result.to_string())
    print(f"Pseudo R-squared: {logit_model.prsquared:.4f}")

    print(f"\n--- SHAP contribution (binary threshold target) ---")
    ranking = shap_contribution_binary(pos_df, features)
    print(ranking.to_string(index=False))

    print(f"\n--- POISSON REGRESSION: raw defensive_contribution count (secondary view) ---")
    _, poisson_result = poisson_regression(pos_df, features)
    print(poisson_result.to_string())


def main():
    df = load_data()
    print(f"Total rows: {len(df)}")
    print(f"Overall hit rate: {df['hit_dc_threshold'].mean():.1%}")

    for position in POSITIONS:
        analyze_position(df, position)

    print("\n" + "=" * 70)
    print("Trust the logistic regression + SHAP results over the Poisson ones "
          "for actual point prediction — the binary threshold is what pays "
          "out, the raw count is context. As always, trust SHAP over "
          "logistic when they disagree on ranking.")


if __name__ == "__main__":
    main()