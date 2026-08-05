"""
FPL Points Predictor — Fixtures Extraction
================================================
Pulls the full fixture list (past and upcoming) from FPL's dedicated
fixtures endpoint, resolves team IDs to names using bootstrap-static,
and separates out the upcoming (unplayed) fixtures specifically — this
is what the prediction pipeline needs to generate real future
predictions instead of only validating against historical gameweeks.

NOTE: I can't reach fantasy.premierleague.com from this sandbox to
verify live field names, same situation as the original bootstrap-static
extraction earlier in this project — written from the documented API
structure, verify against the printed columns on your first real run.

Run:
    python extract_fixtures.py
"""

import logging
from pathlib import Path

import pandas as pd
import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
REQUEST_TIMEOUT = 15

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_team_lookup() -> dict:
    log.info("Fetching team ID -> name lookup from bootstrap-static...")
    r = requests.get(BOOTSTRAP_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    teams = r.json()["teams"]
    return {t["id"]: t["name"] for t in teams}


def get_team_short_name_lookup() -> dict:
    """FPL's own official 3-letter team codes (e.g. 'ARS', 'MUN') —
    same bootstrap-static call, just capturing short_name too."""
    r = requests.get(BOOTSTRAP_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    teams = r.json()["teams"]
    return {t["name"]: t["short_name"] for t in teams}


def get_fixtures() -> list:
    log.info("Fetching fixtures...")
    r = requests.get(FIXTURES_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def build_fixtures_df(fixtures: list, team_lookup: dict, short_name_lookup: dict = None) -> pd.DataFrame:
    short_name_lookup = short_name_lookup or {}
    rows = []
    for f in fixtures:
        home_team = team_lookup.get(f.get("team_h"))
        away_team = team_lookup.get(f.get("team_a"))
        rows.append({
            "fixture_id": f.get("id"),
            "gameweek": f.get("event"),  # None for fixtures not yet scheduled into a gameweek
            "kickoff_time": f.get("kickoff_time"),
            "home_team": home_team,
            "away_team": away_team,
            "home_team_short": short_name_lookup.get(home_team, home_team),
            "away_team_short": short_name_lookup.get(away_team, away_team),
            "home_team_difficulty": f.get("team_h_difficulty"),  # FPL's own 1-5 difficulty rating
            "away_team_difficulty": f.get("team_a_difficulty"),
            "finished": f.get("finished"),
            "started": f.get("started"),
        })
    return pd.DataFrame(rows)


def detect_blank_and_double_gameweeks(upcoming: pd.DataFrame, team_lookup: dict):
    """
    Cup replays and European fixtures regularly force Premier League
    matches to be rearranged, which creates:
      - BLANK gameweeks: a team has ZERO fixtures in a gameweek (their
        players are guaranteed 0 points that week — not poor form,
        just no match played)
      - DOUBLE gameweeks: a team has TWO fixtures in one gameweek (their
        players' points that week are the SUM of both matches)

    Both need explicit handling in the prediction pipeline downstream —
    this just surfaces them clearly every time fixtures are pulled,
    since they can appear or change between runs as fixtures get
    rearranged closer to the date.
    """
    scheduled = upcoming.dropna(subset=["gameweek"])
    if scheduled.empty:
        return

    all_teams = set(team_lookup.values())
    gameweeks = sorted(scheduled["gameweek"].unique())

    print("\n" + "=" * 70)
    print("BLANK / DOUBLE GAMEWEEK CHECK")
    print("=" * 70)

    any_found = False
    for gw in gameweeks:
        gw_fixtures = scheduled[scheduled["gameweek"] == gw]
        teams_playing = pd.concat([gw_fixtures["home_team"], gw_fixtures["away_team"]])
        fixture_counts = teams_playing.value_counts()

        blank_teams = all_teams - set(fixture_counts.index)
        double_teams = fixture_counts[fixture_counts >= 2].index.tolist()

        if blank_teams or double_teams:
            any_found = True
            print(f"\nGameweek {gw:.0f}:")
            if blank_teams:
                print(f"  BLANK for: {sorted(blank_teams)} (0 fixtures — guaranteed 0 points that week)")
            if double_teams:
                print(f"  DOUBLE for: {double_teams} (2 fixtures — sum both matches' predictions)")

    if not any_found:
        print("None detected in the currently scheduled fixtures — but re-check after "
              "re-running this script closer to the date, since rearrangements can "
              "introduce these later even if none exist right now.")


def main():
    team_lookup = get_team_lookup()
    short_name_lookup = get_team_short_name_lookup()
    log.info("Teams found: %d", len(team_lookup))

    fixtures = get_fixtures()
    log.info("Total fixtures: %d", len(fixtures))

    df = build_fixtures_df(fixtures, team_lookup, short_name_lookup)

    all_path = OUTPUT_DIR / "fixtures_all.csv"
    df.to_csv(all_path, index=False)
    print(f"\nSaved all fixtures -> {all_path} ({len(df)} rows)")

    upcoming = df[df["finished"] == False].sort_values("gameweek")  # noqa: E712
    upcoming_path = OUTPUT_DIR / "fixtures_upcoming.csv"
    upcoming.to_csv(upcoming_path, index=False)
    print(f"Saved upcoming fixtures -> {upcoming_path} ({len(upcoming)} rows)")

    print("\n=== Columns ===")
    print(list(df.columns))

    print("\n=== Sample of upcoming fixtures ===")
    print(upcoming.head(10).to_string(index=False))

    if upcoming["gameweek"].isna().any():
        n_unscheduled = upcoming["gameweek"].isna().sum()
        print(f"\nNote: {n_unscheduled} upcoming fixtures have no gameweek assigned yet "
              f"(common for postponed/rearranged matches) — these will need manual handling "
              f"before feeding into the prediction pipeline.")

    detect_blank_and_double_gameweeks(upcoming, team_lookup)


if __name__ == "__main__":
    main()