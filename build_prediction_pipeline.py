"""
FPL Points Predictor — FULL PIPELINE ASSEMBLY
====================================================
Combines every validated component model into one actual points
prediction per player-gameweek, using the real FPL scoring rules
established at the start of this project. This is the final validation
of the whole project: fit everything on early gameweeks, predict later
ones, and see how close the summed prediction lands to actual_points.

COMPONENTS AND HOW EACH IS HANDLED (see each analysis script for full
derivation — this file re-fits lightweight, consistent versions rather
than re-deriving from scratch):

  - Minutes/starts: NOT separately validated before now — a genuine gap
    flagged repeatedly through this project. A simple logistic model is
    fit here (roll5_starts, consecutive_starts, days_since_last_game)
    as a first pass, not a fully explored model like the others.
  - Goals, Assists: Poisson, by position (GK skipped — negligible).
  - Defensive contribution: logistic P(hit threshold), by position.
  - Clean sheets: team-level Poisson on goals conceded, P(0) derived —
    the validated winning approach from build_clean_sheets_model.py.
  - Cards: simple rolling average used directly as the expected count
    (no fitted model — validated earlier that yellow cards have weak
    signal without referee/context data we don't have).
  - Bonus: NOT a real prediction — flagged honestly as unreliable
    (build_bonus_predictions.py showed it doesn't beat baseline). A
    small flat estimate is used so total points aren't systematically
    biased low, not because this component is trustworthy.
  - Saves: Poisson, GK only.

Run:
    python build_prediction_pipeline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75

# FPL scoring rules, position-specific — same rules validated with 0
# mismatches against actual_points at the very start of this project.
GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}  # GK not eligible


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    return df[df["roll5_minutes"].notna()].copy()  # has rolling history


def time_split(df: pd.DataFrame):
    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    return df[df["gameweek"] <= cutoff_gw], df[df["gameweek"] > cutoff_gw], cutoff_gw


def drop_zero_variance(df: pd.DataFrame, features: list) -> list:
    """Shared safety check, applied everywhere a model is fit in this file.
    A zero-variance column causes statsmodels to silently drop it during
    fitting (collinear with the constant), which then breaks prediction
    with a shape mismatch — better to filter it out explicitly upfront
    than hit that downstream."""
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


# =========================
# COMPONENT: minutes thresholds
# =========================
def fit_minutes_threshold_model(train: pd.DataFrame, threshold: int):
    """Fits P(minutes >= threshold). Used twice: threshold=1 for the first
    appearance point (any involvement), threshold=60 for the second
    appearance point AND clean-sheet/defensive-contribution eligibility —
    these are genuinely different FPL rules, not the same probability."""
    features = ["roll5_starts", "consecutive_starts", "days_since_last_game", "roll5_minutes"]
    features = [f for f in features if f in train.columns]
    features = drop_zero_variance(train, features)
    X = sm.add_constant(train[features].fillna(0))
    y = (train["minutes"] >= threshold).astype(int)
    model = sm.Logit(y, X).fit(disp=0)
    return model, features


def predict_p_minutes(model, features, df: pd.DataFrame) -> pd.Series:
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    return pd.Series(model.predict(X), index=df.index)


# =========================
# COMPONENT: goals / assists (Poisson, by position)
# =========================
GOALS_FEATURES = ["season_xG", "roll5_xG", "season_shots", "roll5_shots",
                   "opp_season_shots_for", "was_home_int", "is_primary_pen_taker"]
# possession/PPDA were tested here too but the full pipeline showed no
# aggregate improvement — reverted, unlike DC/CS below where the isolated
# out-of-sample tests proved a real benefit
ASSISTS_FEATURES = ["season_xA", "roll5_xA", "season_key_passes", "roll5_key_passes",
                     "season_Crs", "opp_season_goals_conceded", "was_home_int"]


def fit_poisson_by_position(df: pd.DataFrame, target: str, candidate_features: list, positions: list):
    models = {}
    for position in positions:
        pos_df = df[df["position"] == position]
        features = [f for f in candidate_features if f in pos_df.columns]
        features = drop_zero_variance(pos_df, features)
        if len(pos_df) < 60 or not features:
            continue
        X = sm.add_constant(pos_df[features].fillna(0))
        y = pos_df[target]
        try:
            model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            models[position] = (model, features)
        except Exception as e:
            print(f"  (skipping {target} model for {position}: {e})")
    return models


def predict_poisson(models: dict, df: pd.DataFrame) -> pd.Series:
    preds = pd.Series(0.0, index=df.index)
    for position, (model, features) in models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
        preds.loc[mask] = model.predict(X)
    return preds


# =========================
# COMPONENT: defensive contribution (logistic P(hit threshold), by position)
# =========================
DC_FEATURES = ["season_def_contrib", "roll5_def_contrib", "season_TklW", "season_Int",
               "opp_season_shots_for", "was_home_int",
               # NEW: facing a high-possession opponent means more defensive
               # actions are needed — the most mechanically direct candidate
               # of any of these additions. Own team's pressing intensity
               # (PPDA) also plausibly drives individual defensive-action volume.
               # Validated via analyze_defcontrib_by_position.py's out-of-sample
               # test: DEF Brier 0.164->0.146, MID Brier 0.101->0.088.
               "opp_season_possession", "opp_roll5_possession",
               "own_season_ppda", "own_roll5_ppda"]


def fit_dc_models(df: pd.DataFrame):
    models = {}
    for position in ["DEF", "MID", "FWD"]:
        pos_df = df[df["position"] == position].copy()
        pos_df["hit"] = (pos_df["defensive_contribution"] >= DC_THRESHOLD[position]).astype(int)
        if pos_df["hit"].sum() < 20:
            print(f"  (skipping defensive contribution model for {position}: too few threshold hits)")
            continue
        features = [f for f in DC_FEATURES if f in pos_df.columns]
        features = drop_zero_variance(pos_df, features)
        # proactively prune near-duplicate features (correlation > 0.98) —
        # same fix already proven necessary in build_clean_sheets_model.py.
        # Reactive try/except around the fit alone isn't enough: a
        # regularized fallback can ITSELF fail computing standard errors
        # on a near-singular covariance matrix, as happened here.
        pruned = []
        for f in features:
            too_similar = any(abs(pos_df[f].corr(pos_df[g])) > 0.98 for g in pruned)
            if not too_similar:
                pruned.append(f)
        features = pruned
        X = sm.add_constant(pos_df[features].fillna(0))
        try:
            model = sm.Logit(pos_df["hit"], X).fit(disp=0)
            if not model.mle_retvals.get("converged", True):
                raise np.linalg.LinAlgError("did not converge")
        except np.linalg.LinAlgError as e:
            print(f"  (standard MLE failed for {position} ({e}) — falling back to regularized fit, "
                  f"usually caused by near-perfect separation or high collinearity with the "
                  f"expanded candidate pool)")
            model = sm.Logit(pos_df["hit"], X).fit_regularized(disp=0, alpha=0.1)
        models[position] = (model, features)
    return models


def predict_dc_probability(models: dict, df: pd.DataFrame) -> pd.Series:
    preds = pd.Series(0.0, index=df.index)
    for position, (model, features) in models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
        preds.loc[mask] = model.predict(X)
    return preds


# =========================
# COMPONENT: clean sheets (team-level Poisson on goals conceded)
# =========================
CS_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               # NEW: same additions already being validated in
               # build_clean_sheets_model.py — kept in sync
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]


def fit_clean_sheet_model(df: pd.DataFrame):
    team_df = df.groupby(["team", "fixture_id"]).first().reset_index()
    features = [f for f in CS_FEATURES if f in team_df.columns]
    features = drop_zero_variance(team_df, features)
    X = sm.add_constant(team_df[features].fillna(0))
    y = team_df["goals_conceded"]
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    return model, features


def predict_p_clean_sheet(model, features, df: pd.DataFrame) -> pd.Series:
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    expected_conceded = model.predict(X)
    return pd.Series(np.exp(-expected_conceded), index=df.index)


# =========================
# COMPONENT: saves (GK only, Poisson)
# =========================
SAVES_FEATURES = ["season_saves", "own_season_shots_against", "opp_season_shots_for", "was_home_int"]


def fit_saves_model(df: pd.DataFrame):
    gk_df = df[df["position"] == "GK"]
    features = [f for f in SAVES_FEATURES if f in gk_df.columns]
    features = drop_zero_variance(gk_df, features)
    X = sm.add_constant(gk_df[features].fillna(0))
    y = gk_df["saves"]
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    return model, features


def predict_saves(model, features, df: pd.DataFrame) -> pd.Series:
    mask = df["position"] == "GK"
    preds = pd.Series(0.0, index=df.index)
    if mask.sum() == 0:
        return preds
    X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
    preds.loc[mask] = model.predict(X)
    return preds


# =========================
# ASSEMBLE
# =========================
def compute_avg_bonus_when_played(train: pd.DataFrame) -> float:
    """The bonus component provides ZERO per-player discrimination — every
    player who plays gets the same scaled estimate, since we already proved
    (build_bonus_predictions.py) that predicting WHICH specific player earns
    more bonus doesn't beat baseline. But removing bonus entirely would be
    worse: real players who play DO earn bonus on average (FPL always
    allocates 3+2+1 points to someone every match), so dropping it to zero
    would just shift the bias in the opposite direction for everyone who
    plays. This computes the actual empirical average from training data
    instead of using an arbitrary guessed constant — an honest mean
    correction, clearly not a real prediction."""
    played = train[train["minutes"] >= 60]
    return played["bonus"].mean() if len(played) > 0 else 0.0


def assemble_predictions(test: pd.DataFrame, components: dict, avg_bonus_when_played: float) -> pd.DataFrame:
    df = test.copy()

    df["pred_p_any_minutes"] = predict_p_minutes(*components["any_minutes"], df)
    df["pred_p_60plus"] = predict_p_minutes(*components["sixty_plus"], df)
    df["pred_goals"] = predict_poisson(components["goals"], df)
    df["pred_assists"] = predict_poisson(components["assists"], df)
    df["pred_p_dc_hit"] = predict_dc_probability(components["dc"], df)
    df["pred_p_clean_sheet"] = predict_p_clean_sheet(*components["clean_sheet"], df)
    df["pred_saves"] = predict_saves(*components["saves"], df)
    df["pred_cards"] = df["roll5_CrdY"].fillna(0) if "roll5_CrdY" in df.columns else 0

    # appearance points: correctly two SEPARATE thresholds now, not one
    # probability doing double duty — 1pt for any involvement, +1pt more
    # specifically for reaching 60 minutes
    appearance_pts = df["pred_p_any_minutes"] * 1 + df["pred_p_60plus"] * 1

    # ALL per-match components must be scaled by playing probability, not
    # just clean sheets/bonus — found via check_minutes_calibration.py that
    # the minutes models themselves are well-calibrated, which means the
    # bottom-bucket 12x overprediction was coming from HERE instead: a
    # fringe player with decent historical rolling xG (from occasional past
    # appearances) was getting a full goals/assists/defensive-contribution
    # prediction "as if playing," even in a week they're unlikely to
    # feature at all. Using pred_p_any_minutes as the gate here (not
    # pred_p_60plus) since goals/assists/defensive actions can happen on
    # a substitute appearance too, unlike the clean sheet bonus and second
    # appearance point, which have a genuine 60-minute FPL rule behind them.
    goal_pts = df.apply(lambda r: r["pred_goals"] * GOAL_POINTS.get(r["position"], 0), axis=1) * df["pred_p_any_minutes"]
    assist_pts = df["pred_assists"] * 3 * df["pred_p_any_minutes"]
    dc_pts = df["pred_p_dc_hit"] * 2 * df["pred_p_any_minutes"]
    # clean sheet eligibility is specifically the 60+ minute threshold, not
    # "started at all" — this was the source of an earlier over-prediction
    # bias found in validation (worst for GK/DEF, where clean sheets are
    # worth the most points — exactly the pattern that fix targeted)
    cs_pts = df.apply(
        lambda r: r["pred_p_clean_sheet"] * CLEAN_SHEET_POINTS.get(r["position"], 0) * r["pred_p_60plus"],
        axis=1
    )
    save_pts = (df["pred_saves"] / 3) * df["pred_p_any_minutes"]
    card_pts = -df["pred_cards"] * df["pred_p_any_minutes"]

    # bonus: NOT a real prediction, flagged honestly — but scaled by playing
    # time probability AND calibrated from real training data rather than
    # an arbitrary guessed constant. A flat +0.3 for EVERY player was found
    # (via the bucketed bias diagnostic on the full test set) to badly
    # distort predictions for fringe players who rarely feature. Provides
    # zero per-player discrimination — every player who plays gets the same
    # scaled estimate — but that's an honest reflection of what we actually
    # know here, not a flaw to hide.
    bonus_pts = avg_bonus_when_played * df["pred_p_60plus"]

    df["predicted_points"] = (
        appearance_pts + goal_pts + assist_pts + dc_pts + cs_pts + save_pts + card_pts + bonus_pts
    )
    return df


def main():
    df = load_data()
    train, test, cutoff_gw = time_split(df)
    print(f"Train: {len(train)} rows (gw <= {cutoff_gw:.0f}) | Test: {len(test)} rows (gw > {cutoff_gw:.0f})\n")

    print("Fitting component models on training data...")
    components = {
        "any_minutes": fit_minutes_threshold_model(train, threshold=1),
        "sixty_plus": fit_minutes_threshold_model(train, threshold=60),
        "goals": fit_poisson_by_position(train, "goals", GOALS_FEATURES, ["DEF", "MID", "FWD"]),
        "assists": fit_poisson_by_position(train, "assists", ASSISTS_FEATURES, ["DEF", "MID", "FWD"]),
        "dc": fit_dc_models(train),
        "clean_sheet": fit_clean_sheet_model(train),
        "saves": fit_saves_model(train),
    }

    avg_bonus_when_played = compute_avg_bonus_when_played(train)
    print(f"Calibrated average bonus (players with 60+ minutes): {avg_bonus_when_played:.3f}")

    print("\nGenerating predictions on held-out test data...")
    result = assemble_predictions(test, components, avg_bonus_when_played)

    mae = (result["predicted_points"] - result["actual_points"]).abs().mean()
    signed_bias = (result["predicted_points"] - result["actual_points"]).mean()
    corr = result["predicted_points"].corr(result["actual_points"])
    naive_mae = (train["actual_points"].mean() - result["actual_points"]).abs().mean()

    print(f"\n{'=' * 70}")
    print("PIPELINE VALIDATION — predicted vs actual points, held-out gameweeks")
    print(f"{'=' * 70}")
    print(f"Mean Absolute Error: {mae:.2f} points")
    print(f"Signed bias (mean predicted - actual; 0 = no systematic over/under-prediction): {signed_bias:+.2f} points")
    print(f"Naive baseline MAE (always predict training mean): {naive_mae:.2f} points")
    print(f"Correlation (predicted vs actual): {corr:.3f}")

    print("\nSigned bias by position:")
    for position in ["GK", "DEF", "MID", "FWD"]:
        pos_result = result[result["position"] == position]
        pos_bias = (pos_result["predicted_points"] - pos_result["actual_points"]).mean()
        print(f"  {position}: bias = {pos_bias:+.2f}")

    print("\nBy position:")
    for position in ["GK", "DEF", "MID", "FWD"]:
        pos_result = result[result["position"] == position]
        pos_mae = (pos_result["predicted_points"] - pos_result["actual_points"]).abs().mean()
        print(f"  {position}: MAE = {pos_mae:.2f} (n={len(pos_result)})")

    print("\nBias by predicted-points bucket (uses ALL test rows, not a small sample —")
    print("shows whether the +bias is spread evenly or concentrated at certain prediction levels):")
    result["pred_bucket"] = pd.qcut(result["predicted_points"], q=5, duplicates="drop")
    bucket_stats = result.groupby("pred_bucket", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_predicted": g["predicted_points"].mean(),
            "mean_actual": g["actual_points"].mean(),
            "bias": (g["predicted_points"] - g["actual_points"]).mean(),
        }),
        include_groups=False,
    )
    print(bucket_stats.to_string())

    output_cols = ["player_name", "team", "position", "gameweek", "opponent_team",
                    "predicted_points", "actual_points", "pred_p_any_minutes", "pred_p_60plus",
                    "pred_goals", "pred_assists", "pred_p_dc_hit", "pred_p_clean_sheet", "pred_saves"]
    output_path = PROCESSED_DIR / "pipeline_predictions.csv"
    result[output_cols].to_csv(output_path, index=False)
    print(f"\nSaved full predictions ({len(result)} rows) -> {output_path}")
    print("(Open the CSV directly to inspect individual players — printing a small in-terminal "
          "sample here risks exactly the cherry-picking/anchoring problem it's better to avoid.)")


if __name__ == "__main__":
    main()