import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.platt_bak")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_loader = '''def load_cs_calibrator():
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

new_loader = '''def load_cs_calibrator():
    """Loads the Platt scaling clean sheet calibrator. Switched from
    isotonic after a direct four-way comparison across 6 temporal
    splits found them statistically tied (3/6 wins each, 0.0005 avg
    Brier difference) — Platt preferred as the simpler, smoother
    model: no hard plateaus, no literal 0%/100% predictions, and just
    two plain numbers stored in JSON (no pickle risk at all). Falls
    back gracefully to uncalibrated predictions if the file doesn't
    exist yet - never crashes the pipeline over a missing optional
    file. Returns (coefficient, intercept) or None."""
    calibrator_path = PROCESSED_DIR / "cs_platt_calibrator.json"
    if not calibrator_path.exists():
        print("  (cs_platt_calibrator.json not found - using uncalibrated clean sheet "
              "probabilities. Run fit_and_save_cs_calibrator.py to enable calibration.)")
        return None
    import json
    with open(calibrator_path, "r") as f:
        saved = json.load(f)
    print("  Clean sheet Platt calibrator loaded (validated vs isotonic, statistical tie, "
          "preferred for smoother/safer behavior)")
    return saved["coefficient"], saved["intercept"]'''

already_platt_loader = "Switched from" in content and "isotonic after a direct four-way" in content
if old_loader in content and not already_platt_loader:
    content = content.replace(old_loader, new_loader)
    changes.append("1. Replaced load_cs_calibrator() with Platt JSON loading")
elif already_platt_loader:
    changes.append("1. Already applied - skipped")
else:
    print("*** Change 1 FAILED: isotonic load_cs_calibrator() not found as expected ***")

old_apply = '''    if cs_calibrator is not None:
        calibrated = cs_calibrator["isotonic_model"].predict(df["pred_p_clean_sheet_raw"].values)
        df["pred_p_clean_sheet"] = np.clip(calibrated, cs_calibrator["clip_min"], cs_calibrator["clip_max"])
    else:
        df["pred_p_clean_sheet"] = df["pred_p_clean_sheet_raw"]'''

new_apply = '''    if cs_calibrator is not None:
        _coef, _intercept = cs_calibrator
        _raw_clipped = df["pred_p_clean_sheet_raw"].clip(1e-6, 1 - 1e-6)
        _logit_raw = np.log(_raw_clipped / (1 - _raw_clipped))
        df["pred_p_clean_sheet"] = 1 / (1 + np.exp(-(_coef * _logit_raw + _intercept)))
    else:
        df["pred_p_clean_sheet"] = df["pred_p_clean_sheet_raw"]'''

already_platt_apply = "_coef, _intercept = cs_calibrator" in content
if old_apply in content and not already_platt_apply:
    content = content.replace(old_apply, new_apply)
    changes.append("2. Applied Platt formula instead of isotonic + clip")
elif already_platt_apply:
    changes.append("2. Already applied - skipped")
else:
    print("*** Change 2 FAILED: isotonic application block not found as expected ***")

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
