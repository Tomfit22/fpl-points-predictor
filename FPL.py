import requests
import pandas as pd
import time

# -----------------------------
# CONFIG
# -----------------------------
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
REQUEST_TIMEOUT = 10
SLEEP_TIME = 0.2
MAX_PLAYERS = 200  # set to None once you're ready to pull every player

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
    # -------------------------
    if position in ["GK", "DEF"] and minutes >= 60:
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

        print(f"[{i+1}/{min(MAX_PLAYERS, len(players))}] Processed {player_name}")
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
df.to_csv("fpl_clean_dataset.csv", index=False)

print("\nSaved dataset to: fpl_clean_dataset.csv")
print("\nDataset shape:", df.shape)

# -----------------------------
# QUICK CHECK
# -----------------------------
print("\nPoints comparison:")
print(df[["actual_points", "calculated_points"]].head(10))

mismatches = df[df["actual_points"] != df["calculated_points"]]
print(f"\nRows where calculated points != actual points: {len(mismatches)} / {len(df)}")
if len(mismatches) > 0:
    print(mismatches[["player_name", "gameweek", "actual_points", "calculated_points"]].head(20))