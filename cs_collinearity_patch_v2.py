import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.cscollin_bak2")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_features = '''CS_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               # validated via build_clean_sheets_model.py's out-of-sample
               # test: Brier 0.1947->0.1917 — real but small
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]'''
new_features = '''CS_FEATURES = ["own_roll5_goals_conceded", "own_season_goals_conceded",
               "own_season_xG_against", "opp_season_xG_for",
               "opp_season_goals_scored", "opp_season_shots_for", "was_home_int",
               # validated via build_clean_sheets_model.py's out-of-sample
               # test: Brier 0.1947->0.1917 — real but small
               #
               # Feature ORDER matters — correlation pruning in
               # fit_clean_sheet_model() keeps whichever feature in a
               # correlated pair appears EARLIER here. Confirmed on real
               # data: opp_season_xG_for is a stronger, correctly-signed
               # predictor than opp_season_goals_scored once shots_for is
               # pruned (xG filters out raw-goal-count randomness); the
               # two correlate at 0.689, so they were still fighting each
               # other under the original 0.75 threshold — see
               # diagnose_clean_sheet_opponent_sensitivity.py.
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]'''

if old_features in content:
    content = content.replace(old_features, new_features)
    changes.append("1. Reordered CS_FEATURES to prioritize opp_season_xG_for (evidence-backed)")
elif "Feature ORDER matters" in content:
    changes.append("1. Already applied — skipped")
else:
    print("*** Change 1 FAILED: CS_FEATURES list still not found — diagnostic below ***")
    idx = content.find('CS_FEATURES = [')
    print(repr(content[idx:idx+500]))

old_threshold = '''    pruned = []
    for f in features:
        too_similar = any(abs(team_df[f].corr(team_df[g])) > 0.75 for g in pruned)
        if not too_similar:
            pruned.append(f)
    features = pruned'''
new_threshold = '''    pruned = []
    for f in features:
        # 0.65, not 0.75 — confirmed on real data that opp_season_xG_for
        # and opp_season_goals_scored (0.689 correlation) were still
        # fighting each other for the same signal above 0.75
        too_similar = any(abs(team_df[f].corr(team_df[g])) > 0.65 for g in pruned)
        if not too_similar:
            pruned.append(f)
    features = pruned'''

if old_threshold in content:
    content = content.replace(old_threshold, new_threshold)
    changes.append("2. Lowered pruning threshold from 0.75 to 0.65")
elif "0.65, not 0.75" in content:
    changes.append("2. Already applied — skipped")
else:
    print("*** Change 2 FAILED: pruning threshold block not found as expected ***")

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
