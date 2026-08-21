"""
FPL Points Predictor — Captain Recommendation
========================================================================
Recommends a captain (and vice-captain) from YOUR actual squad for a
given gameweek. Captain scores double points, so the primary criterion
is simply the highest predicted_points among your owned players — but
this also surfaces the CEILING (sim_ceiling from the Monte Carlo
simulation) alongside it, since a genuinely useful captain call is
sometimes "safe reliable pick" vs "high-ceiling differential pick",
not just whichever number is highest on average.

Uses the same --team-file format as analyze_my_team.py, for
consistency (one player name per line, short display names).

Run:
    python recommend_captain.py --team-file my_team.txt
    python recommend_captain.py --team-file my_team.txt --gameweek 3
"""

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def load_team_from_file(path: Path, pool: pd.DataFrame) -> pd.DataFrame:
    raw_names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    pool = pool.copy()
    pool["_norm"] = pool["player_name"].apply(normalize_name)

    matched_rows = []
    unmatched = []
    for name in raw_names:
        norm = normalize_name(name)
        candidates = pool[pool["_norm"] == norm]
        if len(candidates) == 0:
            unmatched.append(name)
        else:
            matched_rows.append(candidates.iloc[0])

    if unmatched:
        print(f"*** {len(unmatched)} name(s) couldn't be matched: {unmatched} ***\n")

    if not matched_rows:
        return pd.DataFrame()
    return pd.DataFrame(matched_rows).drop(columns=["_norm"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-file", type=str, required=True)
    parser.add_argument("--gameweek", type=int, default=None)
    args = parser.parse_args()

    team_file = Path(args.team_file)
    if not team_file.exists():
        print(f"{team_file} not found.")
        return

    path = PROCESSED_DIR / "live_predictions.csv"
    if not path.exists():
        print(f"{path} not found - run build_live_predictions.py first.")
        return

    all_df = pd.read_csv(path)
    gameweek = args.gameweek if args.gameweek is not None else int(all_df["gameweek"].min())
    pool = all_df[all_df["gameweek"] == gameweek].drop_duplicates(subset="player_id").copy()

    my_team = load_team_from_file(team_file, pool)
    if my_team.empty:
        print("No players matched - nothing to recommend.")
        return

    cols = ["player_name", "team", "position", "predicted_points"]
    has_ceiling = "sim_ceiling" in my_team.columns
    has_floor = "sim_floor" in my_team.columns
    if has_ceiling:
        cols.append("sim_ceiling")
    if has_floor:
        cols.append("sim_floor")

    ranked = my_team.sort_values("predicted_points", ascending=False)[cols].reset_index(drop=True)

    print("=" * 70)
    print(f"CAPTAIN RECOMMENDATION - gameweek {gameweek}")
    print("=" * 70)

    top = ranked.iloc[0]
    vice = ranked.iloc[1] if len(ranked) > 1 else None

    print(f"\nCAPTAIN: {top['player_name']} ({top['team']}, {top['position']})")
    print(f"  Predicted points: {top['predicted_points']:.2f} (doubled to {top['predicted_points']*2:.2f} as captain)")
    if has_ceiling:
        print(f"  Ceiling: {top['sim_ceiling']:.2f}" + (f" | Floor: {top['sim_floor']:.2f}" if has_floor else ""))

    if vice is not None:
        print(f"\nVICE-CAPTAIN: {vice['player_name']} ({vice['team']}, {vice['position']})")
        print(f"  Predicted points: {vice['predicted_points']:.2f}")

    if has_ceiling and len(ranked) > 2:
        ceiling_pick = ranked.sort_values("sim_ceiling", ascending=False).iloc[0]
        if ceiling_pick["player_name"] != top["player_name"]:
            print(f"\nWorth considering as a HIGHER-RISK alternative: {ceiling_pick['player_name']} "
                  f"({ceiling_pick['team']}) - lower average ({ceiling_pick['predicted_points']:.2f} pts) "
                  f"but the highest ceiling in your squad ({ceiling_pick['sim_ceiling']:.2f}).")

    print(f"\nFull squad ranked by predicted points:")
    for i, row in ranked.iterrows():
        marker = " <- CAPTAIN" if i == 0 else (" <- VICE" if i == 1 else "")
        print(f"  {i+1}. {row['player_name']:<20} {row['team']:<16} {row['predicted_points']:.2f} pts{marker}")


if __name__ == "__main__":
    main()