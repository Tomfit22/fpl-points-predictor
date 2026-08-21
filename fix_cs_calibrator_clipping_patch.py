import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.cscalib_clip_bak")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_loader = '''def load_cs_calibrator():
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
    return saved["isotonic_model"]'''

new_loader = '''def load_cs_calibrator():
    """Loads the isotonic clean sheet calibrator — validated across 6
    independent temporal splits, 5.9% average Brier improvement, every
    split won (see validate_calibration_multi_split.py). Falls back
    gracefully to uncalibrated predictions if the file doesn't exist
    yet - never crashes the pipeline over a missing optional file.
    Returns the full saved dict (model + clip bounds) — the model is
    intentionally a plain sklearn object, no custom wrapper class,
    since a custom class saved in one script cannot be safely
    unpickled from a different script (confirmed directly — this was
    a real bug in an earlier version of this function)."""
    calibrator_path = PROCESSED_DIR / "cs_isotonic_calibrator.pkl"
    if not calibrator_path.exists():
        print("  (cs_isotonic_calibrator.pkl not found - using uncalibrated clean sheet "
              "probabilities. Run fit_and_save_cs_calibrator.py to enable calibration.)")
        return None
    import pickle
    with open(calibrator_path, "rb") as f:
        saved = pickle.load(f)
    print("  Clean sheet calibrator loaded (validated 6/6 splits, +5.9% avg Brier improvement)")
    return saved'''

already_fixed_loader = "Returns the full saved dict (model + clip bounds)" in content
if old_loader in content and not already_fixed_loader:
    content = content.replace(old_loader, new_loader)
    changes.append("1. Updated load_cs_calibrator() to return the full dict (model + clip bounds)")
elif already_fixed_loader:
    changes.append("1. Already applied - skipped")
else:
    print("*** Change 1 FAILED: load_cs_calibrator() definition not found as expected ***")

old_apply = '''    if cs_calibrator is not None:
        df["pred_p_clean_sheet"] = cs_calibrator.predict(df["pred_p_clean_sheet_raw"].values)
    else:
        df["pred_p_clean_sheet"] = df["pred_p_clean_sheet_raw"]'''

new_apply = '''    if cs_calibrator is not None:
        calibrated = cs_calibrator["isotonic_model"].predict(df["pred_p_clean_sheet_raw"].values)
        df["pred_p_clean_sheet"] = np.clip(calibrated, cs_calibrator["clip_min"], cs_calibrator["clip_max"])
    else:
        df["pred_p_clean_sheet"] = df["pred_p_clean_sheet_raw"]'''

already_clipped = 'cs_calibrator["isotonic_model"].predict' in content
if old_apply in content and not already_clipped:
    content = content.replace(old_apply, new_apply)
    changes.append("2. Applied clipping using the new dict structure")
elif already_clipped:
    changes.append("2. Already applied - skipped")
else:
    print("*** Change 2 FAILED: calibration application block not found as expected ***")

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
