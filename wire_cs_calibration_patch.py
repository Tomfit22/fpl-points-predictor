import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.cscalib_bak")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_def = '''def predict_points(df: pd.DataFrame, components: dict, avg_bonus_by_position: dict, overall_avg_bonus: float, bonus_v2_models: dict) -> pd.DataFrame:'''
new_def = '''def load_cs_calibrator():
    """Loads the isotonic clean sheet calibrator — validated across 6
    independent temporal splits, 5.9% average Brier improvement, every
    split won (see validate_calibration_multi_split.py). Falls back
    gracefully to uncalibrated predictions if the file doesn't exist
    yet - never crashes the pipeline over a missing optional file."""
    calibrator_path = PROCESSED_DIR / "cs_isotonic_calibrator.pkl"
    if not calibrator_path.exists():
        print("  (cs_isotonic_calibrator.pkl not found - using uncalibrated clean sheet "
              "probabilities. Run fit_and_save_cs_calibrator.py to enable calibration.)")
        return None
    import pickle
    with open(calibrator_path, "rb") as f:
        saved = pickle.load(f)
    print("  Clean sheet calibrator loaded (validated 6/6 splits, +5.9% avg Brier improvement)")
    return saved["isotonic_model"]


def predict_points(df: pd.DataFrame, components: dict, avg_bonus_by_position: dict, overall_avg_bonus: float, bonus_v2_models: dict, cs_calibrator=None) -> pd.DataFrame:'''

already_added_loader = "def load_cs_calibrator():" in content
if old_def in content and not already_added_loader:
    content = content.replace(old_def, new_def)
    changes.append("1. Added load_cs_calibrator() and cs_calibrator parameter to predict_points()")
elif already_added_loader:
    changes.append("1. Already applied - skipped")
else:
    print("*** Change 1 FAILED: predict_points() definition not found as expected ***")

old_cs_lines = '''    df["pred_goals_conceded"] = cs_model.predict(X_cs)
    df["pred_p_clean_sheet"] = np.exp(-df["pred_goals_conceded"])'''
new_cs_lines = '''    df["pred_goals_conceded"] = cs_model.predict(X_cs)
    df["pred_p_clean_sheet_raw"] = np.exp(-df["pred_goals_conceded"])
    if cs_calibrator is not None:
        df["pred_p_clean_sheet"] = cs_calibrator.predict(df["pred_p_clean_sheet_raw"].values)
    else:
        df["pred_p_clean_sheet"] = df["pred_p_clean_sheet_raw"]'''

already_calibrated = '"pred_p_clean_sheet_raw"' in content
if old_cs_lines in content and not already_calibrated:
    content = content.replace(old_cs_lines, new_cs_lines)
    changes.append("2. Applied calibration to pred_p_clean_sheet, pred_goals_conceded untouched")
elif already_calibrated:
    changes.append("2. Already applied - skipped")
else:
    print("*** Change 2 FAILED: pred_goals_conceded/pred_p_clean_sheet lines not found as expected ***")

old_call = '''    result = predict_points(fixture_rows, components, avg_bonus_by_position, overall_avg_bonus, bonus_v2_models)'''
new_call = '''    cs_calibrator = load_cs_calibrator()
    result = predict_points(fixture_rows, components, avg_bonus_by_position, overall_avg_bonus, bonus_v2_models, cs_calibrator)'''

already_wired_call = "cs_calibrator = load_cs_calibrator()" in content
if old_call in content and not already_wired_call:
    content = content.replace(old_call, new_call)
    changes.append("3. Wired calibrator loading into the predict_points() call site")
elif already_wired_call:
    changes.append("3. Already applied - skipped")
else:
    print("*** Change 3 FAILED: predict_points() call site not found as expected ***")

old_cols = '''"pred_p_clean_sheet", "pred_goals_conceded", "pred_saves", "pred_pens_saved", "pred_cards", "pred_red_cards",'''
new_cols = '''"pred_p_clean_sheet", "pred_p_clean_sheet_raw", "pred_goals_conceded", "pred_saves", "pred_pens_saved", "pred_cards", "pred_red_cards",'''

already_in_cols = '"pred_p_clean_sheet_raw"' in content.split('component_cols = [')[1][:700] if 'component_cols = [' in content else False
if old_cols in content:
    content = content.replace(old_cols, new_cols)
    changes.append("4. Added pred_p_clean_sheet_raw to the saved column list")
elif already_in_cols:
    changes.append("4. Already applied - skipped")
else:
    print("*** Change 4 FAILED: component_cols list line not found as expected ***")

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
