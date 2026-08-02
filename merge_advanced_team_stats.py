"""
FPL Points Predictor — Merge Advanced Team Stats
======================================================
Combines fbref_team_stats.csv (possession, SoT%, save%, cards) and
understat_match_stats.csv (PPDA, deep completions, xPTS) into one
clean file keyed by (team, match_date) — the same join key used
throughout this project — ready for build_features.py to pick up and
turn into rolling own_/opp_ features, same pattern as everything else.

FBref's team stats file only has game_id, not a date — this pulls
game_id -> date from fbref.read_schedule() (a cheap, separate, already
-cached call, NOT the heavy match-report scraping) to bridge the gap.

Run:
    python merge_advanced_team_stats.py
"""

from pathlib import Path

import pandas as pd
import soccerdata as sd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# fill in once the diagnostic below shows the real mismatched pairs — e.g.
# TEAM_ALIASES = {"Nott'ham Forest": "Nottingham Forest"}
# maps UNDERSTAT's spelling -> FBref's spelling (FBref's is what the rest
# of this project's pipeline already uses as the standard)
TEAM_ALIASES = {
    "Leeds": "Leeds United",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham",
    "Wolverhampton Wanderers": "Wolves",
}


def get_fbref_game_dates() -> pd.DataFrame:
    print("Fetching FBref schedule for game_id -> date mapping (cheap, cached call)...")
    fbref = sd.FBref(leagues="ENG-Premier League", seasons="2025-2026")
    schedule = fbref.read_schedule().reset_index()
    return schedule[["game_id", "date"]].rename(columns={"date": "match_date"})


def load_fbref_team_stats() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "fbref_team_stats.csv")
    dates = get_fbref_game_dates()
    df = df.merge(dates, on="game_id", how="left")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date

    n_missing = df["match_date"].isna().sum()
    if n_missing > 0:
        print(f"*** WARNING: {n_missing} FBref team-stat rows couldn't be matched to a date "
              f"via the schedule lookup — these will be dropped from the merge. ***")
        df = df.dropna(subset=["match_date"])

    keep_cols = ["team", "match_date", "possession", "sot_pct", "saves_pct",
                 "yellow_cards", "red_cards"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].rename(columns={c: f"fb_team_{c}" for c in keep_cols if c not in ["team", "match_date"]})


def load_understat_match_stats() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "understat_match_stats.csv")
    df["match_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    home = df[["team_h", "match_date", "h_ppda", "h_deep", "h_xpts"]].rename(
        columns={"team_h": "team", "h_ppda": "ppda", "h_deep": "deep", "h_xpts": "xpts"})
    away = df[["team_a", "match_date", "a_ppda", "a_deep", "a_xpts"]].rename(
        columns={"team_a": "team", "a_ppda": "ppda", "a_deep": "deep", "a_xpts": "xpts"})

    combined = pd.concat([home, away], ignore_index=True)
    combined["team"] = combined["team"].replace(TEAM_ALIASES)
    return combined.rename(columns={c: f"us_team_{c}" for c in ["ppda", "deep", "xpts"]})


def main():
    fbref = load_fbref_team_stats()
    print(f"FBref team stats: {len(fbref)} rows")

    understat = load_understat_match_stats()
    print(f"Understat match stats: {len(understat)} rows")

    merged = fbref.merge(understat, on=["team", "match_date"], how="outer", indicator=True)
    print(f"\nMerge result: {len(merged)} rows")
    print(merged["_merge"].value_counts().to_string())
    n_matched = (merged["_merge"] == "both").sum()
    print(f"\n{n_matched}/{len(merged)} rows matched on BOTH sources — "
          f"if this is much lower than expected, team name spelling likely "
          f"differs between FBref and Understat (e.g. 'Nott'ham Forest' vs "
          f"'Nottingham Forest') and will need a name-alias fix, same issue "
          f"we handled for the original entity matching months ago.")

    merged = merged.drop(columns=["_merge"])

    fbref_teams = set(fbref["team"].unique())
    understat_teams = set(understat["team"].unique())
    only_in_fbref = fbref_teams - understat_teams
    only_in_understat = understat_teams - fbref_teams
    if only_in_fbref or only_in_understat:
        print(f"\n*** Team names present in FBref but NOT Understat: {sorted(only_in_fbref)} ***")
        print(f"*** Team names present in Understat but NOT FBref: {sorted(only_in_understat)} ***")
        print("These are almost certainly the same teams spelled differently — "
              "match them up above and add to TEAM_ALIASES at the top of this script.")

    output_path = PROCESSED_DIR / "advanced_team_stats.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()