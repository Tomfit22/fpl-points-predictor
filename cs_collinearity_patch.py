import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.cscollin_bak")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_features = '''CS_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]'''
new_features = '''CS_FEATURES = ["own_roll5_goals_conceded", "own_season_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]
# NOTE: order matters here — correlation pruning below keeps whichever
# feature in a correlated pair appears EARLIER in this list. Confirmed
# on real data (see diagnose_clean_sheet_opponent_sensitivity.py):
# opp_season_goals_scored, the single most direct "how dangerous is
# this opponent" signal, was getting diluted to statistical noise
# (coef=-0.02, p=0.83, backwards sign) by collinearity with
# opp_season_xG_for/opp_season_shots_for. Keeping it early and pruning
# the redundant correlated features lets it carry its own real signal.'''

if old_features in content:
    content = content.replace(old_features, new_features)
    changes.append("1. Reordered CS_FEATURES to preserve opp_season_goals_scored during pruning")
elif "NOTE: order matters here" in content:
    changes.append("1. Already applied — skipped")
else:
    print("*** Change 1 FAILED: CS_FEATURES list not found as expected ***")

old_fit = '''def fit_clean_sheet_model(df: pd.DataFrame):
    team_df = df.groupby(["team", "fixture_id"]).first().reset_index()
    features = drop_zero_variance(team_df, [f for f in CS_FEATURES if f in team_df.columns])
    X = sm.add_constant(team_df[features].fillna(0))
    return sm.GLM(team_df["goals_conceded"], X, family=sm.families.Poisson()).fit(), features'''
new_fit = '''def fit_clean_sheet_model(df: pd.DataFrame):
    team_df = df.groupby(["team", "fixture_id"]).first().reset_index()
    features = drop_zero_variance(team_df, [f for f in CS_FEATURES if f in team_df.columns])

    # correlation pruning — confirmed necessary on real data: several
    # CS_FEATURES correlate at 0.71-0.81 (own_season_goals_conceded/
    # own_roll5_goals_conceded, opp_season_xG_for/opp_season_shots_for/
    # opp_season_possession, own_/opp_ possession vs ppda), diluting
    # each feature's individual coefficient even where the combined
    # signal is real — see diagnose_clean_sheet_opponent_sensitivity.py.
    # Threshold of 0.75 here (vs DC's 0.98) since dilution was
    # empirically confirmed happening even at this lower level for
    # this specific model.
    pruned = []
    for f in features:
        too_similar = any(abs(team_df[f].corr(team_df[g])) > 0.75 for g in pruned)
        if not too_similar:
            pruned.append(f)
    features = pruned

    X = sm.add_constant(team_df[features].fillna(0))
    return sm.GLM(team_df["goals_conceded"], X, family=sm.families.Poisson()).fit(), features'''

if old_fit in content:
    content = content.replace(old_fit, new_fit)
    changes.append("2. Added correlation pruning to fit_clean_sheet_model()")
elif "correlation pruning — confirmed necessary on real data" in content:
    changes.append("2. Already applied — skipped")
else:
    print("*** Change 2 FAILED: fit_clean_sheet_model() not found as expected ***")

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
