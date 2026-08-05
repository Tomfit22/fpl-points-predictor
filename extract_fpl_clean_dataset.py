import requests
import pandas as pd
import time
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
REQUEST_TIMEOUT = 10
SLEEP_TIME = 0.2
MAX_PLAYERS = None  # full player set now that sample sizes need to overlap with Understat/FBref
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# POSITION MAP
# -----------------------------
POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

# -----------------------------
# FETCH DATA
# -----------------------------
print("Fetching bootstrap data...")

bootstrap = requests.get(BOOTSTRAP_URL, timeout=REQUEST_TIMEOUT).json()
players = bootstrap["elements"]
teams = {t["id"]: t["name"] for t in bootstrap["teams"]}

print(f"Total players in FPL: {len(players)}")

# -----------------------------
# POINT CALCULATOR
# -----------------------------
def calculate_points(row, position):

    points = 0

    minutes = row["minutes"]
    goals = row["goals_scored"]
    assists = row["assists"]
    cs = row["clean_sheets"]
    saves = row["saves"]
    pens_saved = row["penalties_saved"]
    pens_missed = row["penalties_missed"]
    yc = row["yellow_cards"]
    rc = row["red_cards"]
    own_goals = row["own_goals"]
    def_contrib = row["defensive_contribution"]
    goals_conceded = row["goals_conceded"]
    bonus = row["bonus"]

    # -------------------------
    # Minutes
    # -------------------------
    if minutes > 0:
        points += 1
    if minutes >= 60:
        points += 1

    # -------------------------
    # Goals
    # -------------------------
    if position == "GK":
        points += goals * 10
    elif position == "DEF":
        points += goals * 6
    elif position == "MID":
        points += goals * 5
    elif position == "FWD":
        points += goals * 4

    # -------------------------
    # Assists
    # -------------------------
    points += assists * 3

    # -------------------------
    # Clean Sheets
    # -------------------------
    if minutes >= 60:
        if position in ["GK", "DEF"]:
            points += cs * 4
        elif position == "MID":
            points += cs * 1

    # -------------------------
    # Saves (GK only)
    # -------------------------
    if position == "GK":
        points += (saves // 3) * 1

    # -------------------------
    # Penalties
    # -------------------------
    if position == "GK":
        points += pens_saved * 5
    points -= pens_missed * 2  # FIX: was extracted but never applied

    # -------------------------
    # Own goals
    # -------------------------
    points -= own_goals * 2  # FIX: wasn't captured or scored at all before

    # -------------------------
    # Defensive Contributions
    # FIX: goalkeepers are NOT eligible for defensive contribution points
    # -------------------------
    if position == "DEF":
        if def_contrib >= 10:
            points += 2
    elif position in ["MID", "FWD"]:
        if def_contrib >= 12:
            points += 2

    # -------------------------
    # Cards
    # -------------------------
    points -= yc * 1
    points -= rc * 3

    # -------------------------
    # Goals conceded penalty
    # FIX: this penalty applies to any GK/DEF on the pitch when goals go in,
    # regardless of whether they reached 60 minutes — the 60-min gate only
    # applies to the clean sheet bonus, not this deduction.
    # -------------------------
    if position in ["GK", "DEF"]:
        points -= (goals_conceded // 2)

    # -------------------------
    # BONUS POINTS
    # -------------------------
    points += bonus

    return points

# -----------------------------
# BUILD DATASET
# -----------------------------
rows = []

for i, player in enumerate(players[:MAX_PLAYERS]):

    player_id = player["id"]
    player_name = player["web_name"]
    full_name = f"{player['first_name']} {player['second_name']}"  # better for cross-source name matching
    position = POSITION_MAP[player["element_type"]]
    team = teams[player["team"]]

    try:
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        data = requests.get(url, timeout=REQUEST_TIMEOUT).json()

        history = data.get("history", [])

        for gw in history:

            calculated_points = calculate_points(gw, position)

            rows.append({
                "player_id": player_id,
                "player_name": player_name,
                "full_name": full_name,
                "team": team,
                "position": position,
                "gameweek": gw["round"],

                # FIX: needed to join with Understat/FBref match-level data
                # and to build the opponent-adjusted "points vs a team" features
                "opponent_team": teams[gw["opponent_team"]],
                "was_home": gw["was_home"],
                "kickoff_time": gw["kickoff_time"],
                "fixture_id": gw["fixture"],

                "minutes": gw["minutes"],
                "starts": gw.get("starts"),  # NEW: whether player started (vs sub/unused) — pulled defensively in case the field name differs
                "goals": gw["goals_scored"],
                "assists": gw["assists"],
                "yellow_cards": gw["yellow_cards"],
                "red_cards": gw["red_cards"],
                "own_goals": gw["own_goals"],
                "clean_sheets": gw["clean_sheets"],
                "saves": gw["saves"],
                "penalties_saved": gw["penalties_saved"],
                "penalties_missed": gw["penalties_missed"],
                "defensive_contribution": gw["defensive_contribution"],
                # NEW: granular breakdown, found live on FPL's own official API —
                # zero legal risk (same source as everything else here), zero new
                # scraping infrastructure. Verified: for a DEF, clearances_blocks_
                # interceptions + tackles == defensive_contribution exactly.
                # Also directly feeds BPS as of the 2026/27 rule change (1 BPS per
                # 3 CBI), so these matter beyond just defensive contribution.
                "clearances_blocks_interceptions": gw.get("clearances_blocks_interceptions"),
                "recoveries": gw.get("recoveries"),
                "tackles": gw.get("tackles"),
                "goals_conceded": gw["goals_conceded"],
                "bonus": gw["bonus"],
                "bps": gw["bps"],

                # useful predictor context, already in the API, no extra cost to grab
                "expected_goals": gw.get("expected_goals"),
                "expected_assists": gw.get("expected_assists"),
                "expected_goal_involvements": gw.get("expected_goal_involvements"),
                "expected_goals_conceded": gw.get("expected_goals_conceded"),
                "value": gw["value"],
                "selected": gw["selected"],

                "actual_points": gw["total_points"],
                "calculated_points": calculated_points,
            })

        total_for_print = len(players) if MAX_PLAYERS is None else min(MAX_PLAYERS, len(players))
        print(f"[{i+1}/{total_for_print}] Processed {player_name}")
        time.sleep(SLEEP_TIME)

    except Exception as e:
        print(f"Error with player {player_name}: {e}")

# -----------------------------
# DATAFRAME
# -----------------------------
df = pd.DataFrame(rows)

print("\nDataset created!")
print(df.head())

# -----------------------------
# SAVE
# -----------------------------
output_path = OUTPUT_DIR / "fpl_clean_dataset.csv"

# SAFETY GUARD: refuse to overwrite existing good data with an empty
# result. This happens for real — confirmed on 2026-07-24 — when FPL's
# bootstrap-static resets to a new season's player list before any
# gameweeks have actually been played, meaning every player's history
# is genuinely empty. Silently saving that over a complete prior
# dataset is real, avoidable data loss, not just an inconvenience.
if len(df) == 0:
    if output_path.exists():
        print(f"\n{'*' * 70}")
        print(f"REFUSING TO SAVE: 0 rows collected, but {output_path} already exists "
              f"with real data. This almost always means the new season's player list "
              f"has loaded but no gameweeks have been played yet (per-gameweek history "
              f"is empty for everyone). NOT overwriting your existing data.")
        print(f"Once the new season actually starts, this will work normally again.")
        print(f"{'*' * 70}")
    else:
        print("\n0 rows collected and no existing file to protect — saving empty "
              "result anyway so downstream steps fail loudly rather than silently.")
        df.to_csv(output_path, index=False)
    import sys
    sys.exit(1 if output_path.exists() else 0)

df.to_csv(output_path, index=False)

print(f"\nSaved dataset to: {output_path}")
print("\nDataset shape:", df.shape)

# -----------------------------
# QUICK CHECK
# -----------------------------
# An empty dataset here is EXPECTED and CORRECT during preseason —
# no gameweeks have been played yet, so there's genuinely no
# per-gameweek data to extract. This isn't a failure; the dataset was
# already saved successfully above. Only run the points-comparison
# sanity check when there's actually something to check.
if df.empty or "actual_points" not in df.columns or "calculated_points" not in df.columns:
    print("\n(No gameweek data yet — skipping points-comparison check. "
          "This is expected before the season's first gameweek has been played.)")
else:
    print("\nPoints comparison:")
    print(df[["actual_points", "calculated_points"]].head(10))

    mismatches = df[df["actual_points"] != df["calculated_points"]]
    print(f"\nRows where calculated points != actual points: {len(mismatches)} / {len(df)}")
    if len(mismatches) > 0:
        print(mismatches[["player_name", "gameweek", "actual_points", "calculated_points"]].head(20))