"""
Applies the Phase 1 real-card-model changes to build_live_predictions.py.
Run once from inside FPL_Project:

    python apply_cards_patch.py

Makes a backup first (build_live_predictions.py.bak) and reports exactly
what it changed. Safe to re-run — it checks whether each change is
already applied before touching anything.
"""

import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")


def main():
    if not TARGET.exists():
        print(f"{TARGET} not found — run this from inside FPL_Project.")
        return

    content = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(".py.bak")
    shutil.copy(TARGET, backup)
    print(f"Backed up original to {backup}")

    changes_made = []

    # --- Change 1: extend fit_poisson_by_position with weight_col ---
    old_1 = '''def fit_poisson_by_position(df: pd.DataFrame, target: str, candidate_features: list, positions: list):
    models = {}
    for position in positions:
        pos_df = df[df["position"] == position]
        features = drop_zero_variance(pos_df, [f for f in candidate_features if f in pos_df.columns])
        if len(pos_df) < 60 or not features:
            continue
        X = sm.add_constant(pos_df[features].fillna(0))
        try:
            models[position] = (sm.GLM(pos_df[target], X, family=sm.families.Poisson()).fit(), features)
        except Exception as e:
            print(f"  (skipping {target} model for {position}: {e})")
    return models'''

    new_1 = '''def fit_poisson_by_position(df: pd.DataFrame, target: str, candidate_features: list, positions: list,
                             weight_col: str = None):
    models = {}
    for position in positions:
        pos_df = df[df["position"] == position]
        features = drop_zero_variance(pos_df, [f for f in candidate_features if f in pos_df.columns])
        if len(pos_df) < 60 or not features:
            continue
        X = sm.add_constant(pos_df[features].fillna(0))
        try:
            # freq_weights treats each row as N identical observations —
            # used by cards specifically to weight 25/26 rows more
            # heavily than 24/25 (see build_cards_training_data.py).
            # weight_col=None (default) preserves EXISTING behavior
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
    return models'''

    if old_1 in content:
        content = content.replace(old_1, new_1)
        changes_made.append("1. Extended fit_poisson_by_position with weight_col")
    elif "weight_col: str = None" in content:
        changes_made.append("1. Already applied — skipped")
    else:
        print("*** Change 1 FAILED: exact original text not found. "
              "fit_poisson_by_position may have been edited since this patch was written. ***")

    # --- Change 2: add CARDS_FEATURES near MINUTES_FEATURES ---
    old_2 = 'MINUTES_FEATURES = ["roll5_minutes", "roll5_starts", "consecutive_starts", "days_since_last_game"]'
    new_2 = old_2 + '''
CARDS_FEATURES = ["season_CrdY", "roll5_CrdY", "was_home_int"]
RED_CARDS_FEATURES = ["season_CrdR", "roll5_CrdR", "was_home_int"]'''

    if old_2 in content and "CARDS_FEATURES = " not in content:
        content = content.replace(old_2, new_2)
        changes_made.append("2. Added CARDS_FEATURES/RED_CARDS_FEATURES")
    elif "CARDS_FEATURES = " in content:
        changes_made.append("2. Already applied — skipped")
    else:
        print("*** Change 2 FAILED: MINUTES_FEATURES line not found as expected. ***")

    # --- Change 3: add fit_cards_model() after fit_saves_model's definition ---
    # Inserted right before fit_dc_models so it sits with the other fit_* functions.
    marker_3 = "def fit_dc_models(df: pd.DataFrame):"
    fit_cards_model_code = '''def fit_cards_model(cards_training_path: str = "data/processed/cards_training_combined.csv"):
    """Fits real Poisson models for yellow and red cards, per position,
    on the combined 24/25 + 25/26 weighted training data — replacing
    the naive roll5_CrdY passthrough that previously stood in for an
    actual model. Returns (yellow_models, red_models); either can be
    an empty dict if the training data isn't ready yet, in which case
    the caller falls back to the naive rolling average rather than
    predicting zero for everyone."""
    path = Path(cards_training_path)
    if not path.exists():
        print(f"  {path} not found — run build_cards_training_data.py first. "
              f"Falling back to the naive roll5_CrdY passthrough for now.")
        return {}, {}

    cards_df = pd.read_csv(path)
    positions = ["GK", "DEF", "MID", "FWD"]

    yellow_models = fit_poisson_by_position(
        cards_df, "yellow_cards", CARDS_FEATURES, positions, weight_col="sample_weight"
    )
    red_models = fit_poisson_by_position(
        cards_df, "red_cards", RED_CARDS_FEATURES, positions, weight_col="sample_weight"
    )
    print(f"  Cards model fitted: yellow -> {list(yellow_models.keys())}, "
          f"red -> {list(red_models.keys())}")
    return yellow_models, red_models


'''

    if marker_3 in content and "def fit_cards_model(" not in content:
        content = content.replace(marker_3, fit_cards_model_code + marker_3)
        changes_made.append("3. Added fit_cards_model()")
    elif "def fit_cards_model(" in content:
        changes_made.append("3. Already applied — skipped")
    else:
        print("*** Change 3 FAILED: fit_dc_models marker not found. ***")

    # --- Change 4a: fit cards models + wire into components dict in main() ---
    old_4a = '''        "dc": fit_dc_models(df),
        "clean_sheet": fit_clean_sheet_model(df),
        "saves": fit_saves_model(df),'''
    new_4a = '''        "dc": fit_dc_models(df),
        "clean_sheet": fit_clean_sheet_model(df),
        "saves": fit_saves_model(df),
        "cards_yellow": yellow_card_models,
        "cards_red": red_card_models,'''

    old_4a_setup = "def main():"
    new_4a_setup = '''def main():
    yellow_card_models, red_card_models = fit_cards_model()'''

    already_wired = "yellow_card_models, red_card_models = fit_cards_model()" in content
    if old_4a in content and old_4a_setup in content and not already_wired:
        content = content.replace(old_4a_setup, new_4a_setup, 1)
        content = content.replace(old_4a, new_4a)
        changes_made.append("4a. Wired fit_cards_model() into main() and components dict")
    elif already_wired:
        changes_made.append("4a. Already applied — skipped")
    else:
        print("*** Change 4a FAILED: expected main()/components dict text not found. ***")

    # --- Change 4b: replace the naive passthrough with real predictions ---
    old_4b = '''    df["pred_cards"] = df["roll5_CrdY"].fillna(0) if "roll5_CrdY" in df.columns else 0
    df["pred_red_cards"] = df["roll5_CrdR"].fillna(0) if "roll5_CrdR" in df.columns else 0'''
    new_4b = '''    # real fitted model where available; falls back to the naive rolling
    # average only if fit_cards_model() couldn't fit anything (e.g. the
    # training data hasn't been built yet) — never silently predicts
    # zero across the board.
    df["pred_cards"] = (predict_poisson_by_pos(components["cards_yellow"])
                         if components["cards_yellow"]
                         else (df["roll5_CrdY"].fillna(0) if "roll5_CrdY" in df.columns else 0))
    df["pred_red_cards"] = (predict_poisson_by_pos(components["cards_red"])
                             if components["cards_red"]
                             else (df["roll5_CrdR"].fillna(0) if "roll5_CrdR" in df.columns else 0))'''

    if old_4b in content:
        content = content.replace(old_4b, new_4b)
        changes_made.append("4b. Replaced naive passthrough with real predictions")
    elif "predict_poisson_by_pos(components[\"cards_yellow\"])" in content:
        changes_made.append("4b. Already applied — skipped")
    else:
        print("*** Change 4b FAILED: naive passthrough text not found as expected. ***")

    # --- ensure `from pathlib import Path` exists (fit_cards_model needs it) ---
    if "from pathlib import Path" not in content:
        content = "from pathlib import Path\n" + content
        changes_made.append("(added missing 'from pathlib import Path' import)")

    TARGET.write_text(content, encoding="utf-8")

    print("\n=== Summary ===")
    for c in changes_made:
        print(f"  {c}")
    print(f"\nDone. If anything looks wrong, restore with:\n  cp {backup} {TARGET}")


if __name__ == "__main__":
    main()
