"""
FPL Points Predictor — Team Analyzer & Recommender
========================================================================
Takes YOUR actual 15-man squad and flags real, actionable issues:

  1. TEAM CONCENTRATION RISK — e.g. 3+ players from one real-world team
     facing a run of genuinely tough upcoming fixtures (using the same
     difficulty ratings already in the pipeline). Multiple players from
     one team can crash together in a bad fixture run.

  2. UNDERPERFORMING PICKS — players sitting notably below their own
     position's average predicted points, real transfer-out candidates.

  3. SUGGESTED REPLACEMENTS — for each flagged player, real alternatives
     at the same position, within a realistic price range, genuinely
     better on predicted points — not just "anyone better", but
     affordable, same-position, not-already-owned options.

Input: a simple text file, one player name per line (matches the same
short display names used throughout this project — "web_name").

Run:
    python analyze_my_team.py --team-file my_team.txt
    python analyze_my_team.py --team-file my_team.txt --gameweek 3
"""

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

CONCENTRATION_THRESHOLD_PLAYERS = 3     # 3+ players from one team triggers a check
CONCENTRATION_THRESHOLD_DIFFICULTY = 3.4  # avg upcoming difficulty above this = risky run
FIXTURES_TO_CHECK = 5
UNDERPERFORM_PERCENTILE = 0.35   # bottom 35% of their position = flagged
REPLACEMENT_PRICE_MARGIN = 1.0   # GBP millions of wiggle room above the sold player's price


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
        print(f"*** {len(unmatched)} name(s) from your team file couldn't be matched to "
              f"the current roster — check spelling matches the short display name "
              f"exactly (e.g. 'Haaland', 'B.Fernandes'): {unmatched} ***\n")

    if not matched_rows:
        return pd.DataFrame()
    return pd.DataFrame(matched_rows).drop(columns=["_norm"])


def build_team_fixture_difficulty(full_df: pd.DataFrame, n: int = FIXTURES_TO_CHECK) -> dict:
    """Computes each real-world team's average fixture DIFFICULTY over
    their next N gameweeks, directly from the 'difficulty' column
    already present in live_predictions.csv — deliberately independent
    of the dashboard's own next_fixtures lookup (which only exists
    inside build_dashboard.py's embedded JSON, not as a reusable
    column here)."""
    if "difficulty" not in full_df.columns or "gameweek" not in full_df.columns:
        return {}
    team_fixtures = full_df.drop_duplicates(subset=["team", "gameweek"]).sort_values(["team", "gameweek"])
    lookup = {}
    for team, group in team_fixtures.groupby("team"):
        difficulties = group["difficulty"].head(n).tolist()
        lookup[team] = sum(difficulties) / len(difficulties) if difficulties else None
    return lookup


def check_team_concentration(my_team: pd.DataFrame, pool: pd.DataFrame, team_difficulty: dict):
    print("=" * 70)
    print("TEAM CONCENTRATION RISK")
    print("=" * 70)

    real_team_counts = my_team["team"].value_counts()
    flagged_any = False

    for real_team, count in real_team_counts.items():
        if count < CONCENTRATION_THRESHOLD_PLAYERS:
            continue

        avg_difficulty = team_difficulty.get(real_team)
        players_from_team = my_team[my_team["team"] == real_team]["player_name"].tolist()

        if avg_difficulty is not None and avg_difficulty >= CONCENTRATION_THRESHOLD_DIFFICULTY:
            flagged_any = True
            print(f"\n  [!] {real_team}: you own {count} players ({', '.join(players_from_team)})")
            print(f"    Average upcoming fixture difficulty: {avg_difficulty:.1f}/5 - a genuinely tough run.")
            print(f"    Risk: if {real_team} has a bad week, {count} of your players take the hit together.")
            print(f"    Consider: spreading risk by transferring one of these out for a player at "
                  f"a team with an easier run, or at minimum avoid captaining any of them this week.")
        else:
            print(f"\n  {real_team}: you own {count} players ({', '.join(players_from_team)}) - "
                  f"fixture difficulty looks manageable"
                  f"{f' ({avg_difficulty:.1f}/5)' if avg_difficulty is not None else ''}, no action needed.")

    if not flagged_any and len(real_team_counts[real_team_counts >= CONCENTRATION_THRESHOLD_PLAYERS]) == 0:
        print("\n  No team has 3+ of your players - no concentration risk at all.")


def check_underperformers(my_team: pd.DataFrame, pool: pd.DataFrame):
    print("\n" + "=" * 70)
    print("UNDERPERFORMING PICKS")
    print("=" * 70)

    flagged = []
    for position in ["GK", "DEF", "MID", "FWD"]:
        pos_pool = pool[pool["position"] == position]
        if pos_pool.empty:
            continue
        threshold = pos_pool["predicted_points"].quantile(UNDERPERFORM_PERCENTILE)

        my_pos_players = my_team[my_team["position"] == position]
        for _, player in my_pos_players.iterrows():
            if player["predicted_points"] < threshold:
                flagged.append((player, threshold))

    if not flagged:
        print("\n  No picks flagged - your whole squad is performing reasonably for their positions.")
        return []

    for player, threshold in flagged:
        print(f"\n  [!] {player['player_name']} ({player['team']}, {player['position']}): "
              f"predicted {player['predicted_points']:.2f} pts, "
              f"below the bottom {int(UNDERPERFORM_PERCENTILE*100)}% threshold for {player['position']} "
              f"({threshold:.2f} pts)")
    return flagged


def suggest_replacements(flagged: list, my_team: pd.DataFrame, pool: pd.DataFrame):
    if not flagged:
        return
    print("\n" + "=" * 70)
    print("SUGGESTED REPLACEMENTS")
    print("=" * 70)

    owned_ids = set(my_team["player_id"])

    for player, _ in flagged:
        position = player["position"]
        price = player["value"] / 10.0 if "value" in player else None
        candidates = pool[(pool["position"] == position) & (~pool["player_id"].isin(owned_ids))].copy()
        if price is not None and "value" in candidates.columns:
            candidates["price"] = candidates["value"] / 10.0
            candidates = candidates[candidates["price"] <= price + REPLACEMENT_PRICE_MARGIN]
        candidates = candidates[candidates["predicted_points"] > player["predicted_points"]]
        candidates = candidates.sort_values("predicted_points", ascending=False).head(3)

        print(f"\n  For {player['player_name']} (currently {player['predicted_points']:.2f} pts):")
        if candidates.empty:
            print("    No clearly better, affordable alternative found at this price point.")
        else:
            for _, cand in candidates.iterrows():
                cand_price = cand["value"] / 10.0 if "value" in cand else None
                price_str = f"GBP{cand_price:.1f}m" if cand_price is not None else "price n/a"
                print(f"    -> {cand['player_name']:<20} {cand['team']:<16} {price_str}  "
                      f"{cand['predicted_points']:.2f} pts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-file", type=str, required=True, help="Text file, one player name per line")
    parser.add_argument("--gameweek", type=int, default=None, help="Gameweek to analyze (default: earliest available)")
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

    # fixture difficulty lookup uses the FULL multi-gameweek data (all
    # upcoming gameweeks), not just the single filtered gameweek above
    team_difficulty = build_team_fixture_difficulty(all_df[all_df["gameweek"] >= gameweek])

    my_team = load_team_from_file(team_file, pool)
    if my_team.empty:
        print("No players from your team file could be matched - nothing to analyze.")
        return

    print(f"Loaded {len(my_team)} of your players for gameweek {gameweek}\n")

    check_team_concentration(my_team, pool, team_difficulty)
    flagged = check_underperformers(my_team, pool)
    suggest_replacements(flagged, my_team, pool)


if __name__ == "__main__":
    main()