import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.csrolling_bak")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_features = '''CS_FEATURES = ["own_roll5_goals_conceded", "own_season_goals_conceded",
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
new_features = '''CS_FEATURES = ["own_roll5_goals_conceded", "own_season_goals_conceded",
               "own_roll5_xG_against", "own_season_xG_against",
               "opp_roll5_xG_for", "opp_season_xG_for",
               "opp_roll5_goals_scored", "opp_season_goals_scored",
               "opp_season_shots_for", "was_home_int",
               # validated via build_clean_sheets_model.py's out-of-sample
               # test: Brier 0.1947->0.1917 — real but small
               #
               # Feature ORDER matters — correlation pruning in
               # fit_clean_sheet_model() keeps whichever feature in a
               # correlated pair appears EARLIER here. ROLLING (roll5)
               # versions placed before their SEASON-AVERAGE counterparts
               # — season averages are slow-moving and get tangled up
               # with other season-level confounds (confirmed on real
               # data: even a correctly-signed, significant coefficient
               # on opp_season_xG_for only produced an ~9% real-world
               # difference for facing elite vs weak opponents, despite
               # the underlying feature itself differing by ~38% — the
               # net effect was diluted by other season-level features
               # moving in partially offsetting directions for this
               # specific subset of matches). Rolling recent-form
               # features are more directly fixture-relevant and don't
               # carry the same full-season smoothing.
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]'''

if old_features in content:
    content = content.replace(old_features, new_features)
    changes.append("1. Added rolling opponent/own-team features as prioritized candidates")
elif "ROLLING (roll5)" in content:
    changes.append("1. Already applied — skipped")
else:
    print("*** Change 1 FAILED: CS_FEATURES list not found — diagnostic below ***")
    idx = content.find('CS_FEATURES = [')
    print(repr(content[idx:idx+600]))

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
