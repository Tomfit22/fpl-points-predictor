"""
FPL Points Predictor — SAVES (goalkeepers)
================================================
Unlike clean sheets (team-level, binary), saves are a genuinely
individual, count-based GK stat — so back to Poisson regression, same
family as goals/assists.

The key driver should be shot volume FACED, which is really a team-
level quantity (how many shots the opponent takes against this GK's
team) — we already have this via own_roll5_shots_against /
opp_roll5_shots_for (both represent the same underlying thing from
different angles: shots this team's defense allows). A GK on a leaky
defense facing lots of shots gets MORE save opportunities, which is
actually a silver lining for their FPL saves points even though it
means their team is defensively weak.

Also includes each GK's own rolling save RATE (season_saves /
season_shots_against, if computable) as a proxy for shot-stopping
quality, distinct from pure opportunity volume.

Run:
    python build_saves_model.py
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
    "season_saves", "roll5_saves",
    "own_season_shots_against", "own_roll5_shots_against",
    "opp_season_shots_for", "opp_roll5_shots_for",
    "own_season_xG_against", "own_roll5_xG_against",
    "opp_season_xG_for", "opp_roll5_xG_for",
    "roll5_minutes", "roll5_starts",
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
    df = df[(df["position"] == "GK") & (df["minutes"] > 0) & (df["roll5_saves"].notna())].copy()
    # a GK's own historical save RATE (saves per shot faced) — a quality
    # proxy distinct from pure shot-volume opportunity. Guards against
    # divide-by-zero for GKs with no rolling shots-against history yet.
    if "season_saves" in df.columns and "own_season_shots_against" in df.columns:
        df["season_save_rate"] = (df["season_saves"] / df["own_season_shots_against"].replace(0, np.nan))
    return df


def select_features(df: pd.DataFrame) -> list:
    pool = CANDIDATE_POOL + (["season_save_rate"] if "season_save_rate" in df.columns else [])
    available = [f for f in pool if f in df.columns]

    variances = df[available].var(numeric_only=True)
    zero_var_cols = variances[variances.fillna(0) == 0].index.tolist()
    available = [f for f in available if f not in zero_var_cols]

    corrs = df[available + ["saves"]].corr(numeric_only=True)["saves"].drop("saves").abs()
    corr_df = corrs.reset_index()
    corr_df.columns = ["feature", "abs_corr"]
    corr_df["family"] = corr_df["feature"].apply(get_feature_family)
    best_per_family = corr_df.sort_values("abs_corr", ascending=False).drop_duplicates(subset="family", keep="first")
    return best_per_family.sort_values("abs_corr", ascending=False)["feature"].tolist()


def poisson_regression(df: pd.DataFrame, features: list):
    X = sm.add_constant(df[features].fillna(0))
    y = df["saves"]
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
    y = df["saves"]

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


def evaluate_out_of_sample(df: pd.DataFrame, features: list):
    """Time-based split, WITH a real baseline (naive = always predict the
    training mean) — same lesson from bonus points, applied from the start."""
    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw]

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(train[features].fillna(0), train["saves"])
    preds = rf.predict(test[features].fillna(0))

    model_mae = np.abs(preds - test["saves"]).mean()
    naive_pred = train["saves"].mean()
    naive_mae = np.abs(naive_pred - test["saves"]).mean()

    print(f"\nTrain: {len(train)} rows (gw <= {cutoff_gw:.0f}) | Test: {len(test)} rows (gw > {cutoff_gw:.0f})")
    print(f"Naive baseline (always predict training mean of {naive_pred:.2f} saves): MAE = {naive_mae:.2f}")
    print(f"Model MAE: {model_mae:.2f}")
    if model_mae >= naive_mae:
        print("*** WARNING: model does NOT beat the naive mean-prediction baseline. ***")
    else:
        print(f"Model beats the naive baseline by {naive_mae - model_mae:.2f} saves of MAE.")


def main():
    df = load_data()
    print(f"Total GK rows: {len(df)}")
    print(f"Mean saves per match: {df['saves'].mean():.2f} | Std: {df['saves'].std():.2f}")

    features = select_features(df)
    print(f"\nFeatures selected (one per family): {len(features)}")
    for f in features:
        print(f"  {f}  [family: {get_feature_family(f)}]")

    print(f"\n--- POISSON REGRESSION: predicting saves ---")
    model, result = poisson_regression(df, features)
    print(result.to_string())
    print(f"Pseudo R-squared: {model.pseudo_rsquared():.4f}")

    print(f"\n--- SHAP contribution ---")
    ranking, _ = shap_contribution(df, features)
    print(ranking.to_string(index=False))

    print(f"\n--- OUT-OF-SAMPLE VALIDATION ---")
    evaluate_out_of_sample(df, features)


if __name__ == "__main__":
    main()