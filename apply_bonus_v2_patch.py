"""
Applies the validated per-player bonus model (bonus v2) to
build_live_predictions.py, replacing the flat position-average with
real per-player predictions from fit_poisson_by_position — validated
to beat the baseline by 13.9% MAE on genuinely held-out data (see
validate_bonus_v2.py's output).

Falls back to the existing per-position average for any player missing
the required features (e.g. genuinely new signings with no rolling
history yet) — never silently predicts zero or crashes.

Run once from inside FPL_Project:

    python apply_bonus_v2_patch.py

Makes a backup first and is safe to re-run.
"""

import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")


def main():
    if not TARGET.exists():
        print(f"{TARGET} not found — run this from inside FPL_Project.")
        return

    content = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(".py.bonusv2_bak")
    shutil.copy(TARGET, backup)
    print(f"Backed up original to {backup}")

    changes_made = []

    # --- Change 1: add BONUS_FEATURES near the other _FEATURES lists ---
    marker_1 = 'RED_CARDS_FEATURES = ["season_CrdR", "roll5_CrdR", "was_home_int"]'
    new_1 = marker_1 + '''
BONUS_FEATURES = ["roll5_bps", "season_bps", "roll5_CrdY", "roll5_minutes", "was_home_int"]'''

    if marker_1 in content and "BONUS_FEATURES = " not in content:
        content = content.replace(marker_1, new_1)
        changes_made.append("1. Added BONUS_FEATURES")
    elif "BONUS_FEATURES = " in content:
        changes_made.append("1. Already applied — skipped")
    else:
        print("*** Change 1 FAILED: RED_CARDS_FEATURES marker not found. "
              "Run apply_cards_patch.py first if you haven't already. ***")

    # --- Change 2: add fit_bonus_v2_model(), right before fit_clean_sheet_model() ---
    marker_2 = "def fit_clean_sheet_model(df: pd.DataFrame):"
    fit_bonus_v2_code = '''def fit_bonus_v2_model(bonus_training_path: str = "data/processed/bonus_training_combined.csv"):
    """Fits a real per-player expected-bonus model (Poisson per
    position) — validated on genuinely held-out data to beat the flat
    position-average baseline by 13.9% MAE (see validate_bonus_v2.py).
    Returns an empty dict if the training data isn't ready, in which
    case the caller falls back to the per-position average rather than
    predicting zero for everyone."""
    path = Path(bonus_training_path)
    if not path.exists():
        print(f"  {path} not found — run build_bonus_training_data.py first. "
              f"Falling back to the per-position bonus average for now.")
        return {}

    bonus_df = pd.read_csv(path)
    positions = ["GK", "DEF", "MID", "FWD"]
    models = fit_poisson_by_position(
        bonus_df, "bonus", BONUS_FEATURES, positions, weight_col="sample_weight"
    )
    print(f"  Bonus v2 model fitted: {list(models.keys())}")
    return models


'''

    if marker_2 in content and "def fit_bonus_v2_model(" not in content:
        content = content.replace(marker_2, fit_bonus_v2_code + marker_2)
        changes_made.append("2. Added fit_bonus_v2_model()")
    elif "def fit_bonus_v2_model(" in content:
        changes_made.append("2. Already applied — skipped")
    else:
        print("*** Change 2 FAILED: fit_clean_sheet_model marker not found. ***")

    # --- Change 3: fit the model in main(), alongside cards ---
    old_3 = "    yellow_card_models, red_card_models = fit_cards_model()"
    new_3 = old_3 + "\n    bonus_v2_models = fit_bonus_v2_model()"

    if old_3 in content and "bonus_v2_models = fit_bonus_v2_model()" not in content:
        content = content.replace(old_3, new_3)
        changes_made.append("3. Fitted bonus_v2_models in main()")
    elif "bonus_v2_models = fit_bonus_v2_model()" in content:
        changes_made.append("3. Already applied — skipped")
    else:
        print("*** Change 3 FAILED: fit_cards_model() call site not found. "
              "Run apply_cards_patch.py first if you haven't already. ***")

    # --- Change 4: replace flat position bonus_pts with per-player v2 predictions ---
    old_4a = '    bonus_pts = df["position"].map(avg_bonus_by_position).fillna(overall_avg_bonus) * df["pred_p_60plus"]'
    new_4a = '''    # per-player bonus prediction where available (validated 13.9% MAE
    # improvement over the flat position average — see
    # validate_bonus_v2.py); falls back to the position average for any
    # player missing the required rolling features (e.g. brand-new
    # signings with no history yet), never silently zero
    position_fallback_bonus = df["position"].map(avg_bonus_by_position).fillna(overall_avg_bonus)
    if bonus_v2_models:
        v2_bonus_pred = predict_poisson_by_pos(bonus_v2_models)
        per_player_bonus = v2_bonus_pred.where(v2_bonus_pred.notna() & (v2_bonus_pred > 0), position_fallback_bonus)
    else:
        per_player_bonus = position_fallback_bonus
    bonus_pts = per_player_bonus * df["pred_p_60plus"]'''

    if old_4a in content:
        content = content.replace(old_4a, new_4a)
        changes_made.append("4. Wired per-player bonus v2 into predict_points()")
    elif "per_player_bonus = v2_bonus_pred.where" in content:
        changes_made.append("4. Already applied — skipped")
    else:
        print("*** Change 4 FAILED: predict_points() bonus_pts line not found as expected. ***")

    # --- Change 5: pass bonus_v2_models into predict_points() call site ---
    old_5 = "    result = predict_points(fixture_rows, components, avg_bonus_by_position, overall_avg_bonus)"
    new_5 = "    result = predict_points(fixture_rows, components, avg_bonus_by_position, overall_avg_bonus, bonus_v2_models)"

    old_5_sig = "def predict_points(df: pd.DataFrame, components: dict, avg_bonus_by_position: dict, overall_avg_bonus: float) -> pd.DataFrame:"
    new_5_sig = "def predict_points(df: pd.DataFrame, components: dict, avg_bonus_by_position: dict, overall_avg_bonus: float, bonus_v2_models: dict) -> pd.DataFrame:"

    if old_5 in content and old_5_sig in content:
        content = content.replace(old_5, new_5)
        content = content.replace(old_5_sig, new_5_sig)
        changes_made.append("5. Updated predict_points() signature and call site")
    elif "bonus_v2_models: dict) -> pd.DataFrame:" in content:
        changes_made.append("5. Already applied — skipped")
    else:
        print("*** Change 5 FAILED: predict_points() signature/call site not found as expected. ***")

    TARGET.write_text(content, encoding="utf-8")

    print("\n=== Summary ===")
    for c in changes_made:
        print(f"  {c}")
    print(f"\nNote: Monte Carlo simulation intentionally left using the position\n"
          f"average, not bonus v2 — the simulation needs a real per-player\n"
          f"VARIANCE, not just a point estimate, which this model doesn't\n"
          f"provide. Using v2's point estimate there would silently misstate\n"
          f"the simulated uncertainty. predicted_points itself (the main\n"
          f"number) DOES use the improved v2 prediction via predict_points().")
    print(f"\nDone. If anything looks wrong, restore with:\n  cp {backup} {TARGET}")


if __name__ == "__main__":
    main()
