import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.lowconf_bak")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_early_return = '''    if not cold_start_mask.any():
        return player_snapshot.drop(columns=["_games_played", "_price_ratio"])'''
new_early_return = '''    player_snapshot["is_cold_start"] = cold_start_mask

    if not cold_start_mask.any():
        return player_snapshot.drop(columns=["_games_played", "_price_ratio"])'''

already_marked_cold_start = 'player_snapshot["is_cold_start"] = cold_start_mask' in content
if old_early_return in content and not already_marked_cold_start:
    content = content.replace(old_early_return, new_early_return)
    changes.append("1. Added is_cold_start column")
elif already_marked_cold_start:
    changes.append("1. Already applied — skipped")
else:
    print("*** Change 1 FAILED: early-return block not found as expected ***")

old_fallback_end = '''    if teams_using_fallback:
        print(f"\\n*** {len(teams_using_fallback)} team(s) have zero PL history (newly promoted) — "
              f"using league-average stats as a fallback: {sorted(teams_using_fallback)}. "
              f"Treat predictions involving these teams as lower-confidence. ***")

    if not rows:'''
new_fallback_end = '''    if teams_using_fallback:
        print(f"\\n*** {len(teams_using_fallback)} team(s) have zero PL history (newly promoted) — "
              f"using league-average stats as a fallback: {sorted(teams_using_fallback)}. "
              f"Treat predictions involving these teams as lower-confidence. ***")

    # mark every row where EITHER side of the fixture used the
    # league-average fallback — a real, known reason to trust this
    # specific prediction less, now carried through to the final output
    # instead of only appearing as a console message
    for team_players in rows:
        team_players["uses_fallback_stats"] = (
            team_players["team"].isin(teams_using_fallback)
            | team_players["opponent_team"].isin(teams_using_fallback)
        )

    if not rows:'''

already_marked_fallback = '"uses_fallback_stats"' in content
if old_fallback_end in content and not already_marked_fallback:
    content = content.replace(old_fallback_end, new_fallback_end)
    changes.append("2. Added uses_fallback_stats column to fixture rows")
elif already_marked_fallback:
    changes.append("2. Already applied — skipped")
else:
    print("*** Change 2 FAILED: teams_using_fallback block not found as expected ***")

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
