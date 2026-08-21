import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.lowconf_wiring_bak2")
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

already_combined = 'result["is_low_confidence"] = (' in content
if old_predict in content and not already_combined:
    content = content.replace(old_predict, new_predict)
    changes.append("1. Combined is_cold_start + uses_fallback_stats into is_low_confidence")
elif already_combined:
    changes.append("1. Already applied — skipped")
else:
    print("*** Change 1 FAILED: predict_points() call site still not found — diagnostic below ***")
    idx = content.find('result = predict_points(fixture_rows')
    print(repr(content[idx:idx+250]))

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
