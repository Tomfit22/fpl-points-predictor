import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.lowconf_wiring_bak")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_predict = '''    result = predict_points(fixture_rows, components, avg_bonus_by_position, overall_avg_bonus, bonus_v2_models)
    print("Running Monte Carlo simulation (5,000 draws per player) for prediction ranges...")'''
new_predict = '''    result = predict_points(fixture_rows, components, avg_bonus_by_position, overall_avg_bonus, bonus_v2_models)

    # combined low-confidence flag — either a cold-start player (new
    # signing, no real PL history to predict from) or a fixture
    # involving a promoted team's league-average fallback stats. Both
    # are real, known reasons to trust this specific prediction less;
    # this makes that visible downstream (e.g. the dashboard) instead
    # of only ever appearing as a console message during generation.
    result["is_low_confidence"] = (
        result.get("is_cold_start", False).fillna(False).astype(bool)
        if "is_cold_start" in result.columns else False
    ) | (
        result.get("uses_fallback_stats", False).fillna(False).astype(bool)
        if "uses_fallback_stats" in result.columns else False
    )

    print("Running Monte Carlo simulation (5,000 draws per player) for prediction ranges...")'''

already_combined = '"is_low_confidence"' in content
if old_predict in content and not already_combined:
    content = content.replace(old_predict, new_predict)
    changes.append("1. Combined is_cold_start + uses_fallback_stats into is_low_confidence")
elif already_combined:
    changes.append("1. Already applied — skipped")
else:
    print("*** Change 1 FAILED: predict_points() call site not found as expected ***")

old_cols = '''    component_cols = ["predicted_points", "pred_goals", "pred_assists", "pred_p_dc_hit",
                       "pred_p_clean_sheet", "pred_goals_conceded", "pred_saves", "pred_pens_saved", "pred_cards", "pred_red_cards",
                       "pred_p_any_minutes", "pred_p_60plus",
                       "pts_appearance", "pts_goals", "pts_assists", "pts_dc", "pts_clean_sheet",
                       "pts_saves", "pts_pen_saves", "pts_cards", "pts_gc_penalty", "pts_bonus",
                       "sim_floor", "sim_p25", "sim_median", "sim_p75", "sim_ceiling"]'''
new_cols = '''    component_cols = ["predicted_points", "pred_goals", "pred_assists", "pred_p_dc_hit",
                       "pred_p_clean_sheet", "pred_goals_conceded", "pred_saves", "pred_pens_saved", "pred_cards", "pred_red_cards",
                       "pred_p_any_minutes", "pred_p_60plus", "is_low_confidence",
                       "pts_appearance", "pts_goals", "pts_assists", "pts_dc", "pts_clean_sheet",
                       "pts_saves", "pts_pen_saves", "pts_cards", "pts_gc_penalty", "pts_bonus",
                       "sim_floor", "sim_p25", "sim_median", "sim_p75", "sim_ceiling"]'''

already_in_cols = '"is_low_confidence"' in content.split('component_cols = [')[1][:700] if 'component_cols = [' in content else False
if old_cols in content:
    content = content.replace(old_cols, new_cols)
    changes.append("2. Added is_low_confidence to the saved column list")
elif already_in_cols:
    changes.append("2. Already applied — skipped")
else:
    print("*** Change 2 FAILED: component_cols list not found as expected ***")

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
