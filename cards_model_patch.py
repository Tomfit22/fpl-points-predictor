"""
Patch for build_live_predictions.py — Real Card Model (Phase 1)
========================================================================
Replaces the naive roll5_CrdY/roll5_CrdR passthrough (previously just
using the player's own rolling average directly as the "prediction",
with no actual fitted model at all) with a genuine Poisson model per
position, trained on the combined 24/25 + 25/26 dataset built by
build_cards_training_data.py.

Apply these changes to build_live_predictions.py:

1. Extend fit_poisson_by_position() to optionally accept sample
   weights — existing goals/assists calls are UNAFFECTED (weight_col
   defaults to None, preserving current behavior exactly).

2. Add CARDS_FEATURES and a fit_cards_model() wrapper.

3. Replace the naive passthrough at the point predictions are made.
"""

# =========================================================================
# CHANGE 1 — extend fit_poisson_by_position with optional sample weights
# =========================================================================
# Replace the existing fit_poisson_by_position with this version:

def fit_poisson_by_position(df, target, candidate_features, positions, weight_col=None):
    models = {}
    for position in positions:
        pos_df = df[df["position"] == position]
        features = drop_zero_variance(pos_df, [f for f in candidate_features if f in pos_df.columns])
        if len(pos_df) < 60 or not features:
            continue
        X = sm.add_constant(pos_df[features].fillna(0))
        try:
            # freq_weights treats each row as representing N identical
            # observations — a reasonable fit for "this 25/26 row
            # should count 3x as much as a 24/25 row" (see
            # build_cards_training_data.py for the weighting rationale).
            # weight_col=None (the default) preserves EXISTING behavior
            # exactly for goals/assists, which don't use this.
            fit_kwargs = {}
            if weight_col and weight_col in pos_df.columns:
                fit_kwargs["freq_weights"] = pos_df[weight_col]
            models[position] = (
                sm.GLM(pos_df[target], X, family=sm.families.Poisson(), **fit_kwargs).fit(),
                features,
            )
        except Exception as e:
            print(f"  (skipping {target} model for {position}: {e})")
    return models


# =========================================================================
# CHANGE 2 — add near the other _FEATURES lists (after MINUTES_FEATURES)
# =========================================================================

CARDS_FEATURES = ["season_CrdY", "roll5_CrdY", "was_home_int"]
RED_CARDS_FEATURES = ["season_CrdR", "roll5_CrdR", "was_home_int"]


def fit_cards_model(cards_training_path="data/processed/cards_training_combined.csv"):
    """Fits real Poisson models for yellow and red cards, per position,
    on the combined 24/25 + 25/26 training data — replacing the naive
    roll5_CrdY passthrough that was standing in for an actual model.

    Returns (yellow_models, red_models), each a dict keyed by position,
    same structure fit_poisson_by_position already returns elsewhere —
    or (None, None) if the training data hasn't been built yet, so the
    caller can fall back to the naive approach rather than crash.
    """
    import pandas as pd
    from pathlib import Path

    path = Path(cards_training_path)
    if not path.exists():
        print(f"  {path} not found — run build_cards_training_data.py first. "
              f"Falling back to the naive roll5_CrdY passthrough for now.")
        return None, None

    df = pd.read_csv(path)
    positions = ["GK", "DEF", "MID", "FWD"]

    yellow_models = fit_poisson_by_position(
        df, "yellow_cards", CARDS_FEATURES, positions, weight_col="sample_weight"
    )
    red_models = fit_poisson_by_position(
        df, "red_cards", RED_CARDS_FEATURES, positions, weight_col="sample_weight"
    )

    print(f"  Cards model fitted: yellow -> {list(yellow_models.keys())}, "
          f"red -> {list(red_models.keys())}")
    return yellow_models, red_models


# =========================================================================
# CHANGE 3 — replace the naive passthrough with real predictions
# =========================================================================
# Find this block (around line 586-587):
#
#     df["pred_cards"] = df["roll5_CrdY"].fillna(0) if "roll5_CrdY" in df.columns else 0
#     df["pred_red_cards"] = df["roll5_CrdR"].fillna(0) if "roll5_CrdR" in df.columns else 0
#
# Replace with a call to predict_cards() (defined below).

def predict_cards(df, yellow_models, red_models):
    """Applies fitted per-position Poisson models to generate real card
    predictions — falls back to the naive rolling average ONLY for a
    position with no fitted model (e.g. too few observations), never
    silently for everyone."""
    df = df.copy()
    df["pred_cards"] = df["roll5_CrdY"].fillna(0) if "roll5_CrdY" in df.columns else 0.0
    df["pred_red_cards"] = df["roll5_CrdR"].fillna(0) if "roll5_CrdR" in df.columns else 0.0

    if yellow_models:
        for position, (model, features) in yellow_models.items():
            mask = df["position"] == position
            if mask.sum() == 0:
                continue
            X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
            X = X.reindex(columns=model.params.index, fill_value=0)  # match fitted column order
            df.loc[mask, "pred_cards"] = model.predict(X)

    if red_models:
        for position, (model, features) in red_models.items():
            mask = df["position"] == position
            if mask.sum() == 0:
                continue
            X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
            X = X.reindex(columns=model.params.index, fill_value=0)
            df.loc[mask, "pred_red_cards"] = model.predict(X)

    return df


# =========================================================================
# CHANGE 4 — wire it into main(), near where other models get fitted
# =========================================================================
# Add alongside the other fit_*_model() calls in main():
#
#     yellow_card_models, red_card_models = fit_cards_model()
#
# Then replace the naive passthrough lines with:
#
#     df = predict_cards(df, yellow_card_models, red_card_models)