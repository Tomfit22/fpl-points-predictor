"""
FPL Points Predictor — Understat Extraction (v2)
==================================================
Replaces the soccerdata-based Understat pull, which depended on a
compiled TLS-fingerprinting binary (tls_requests) that failed to load
on macOS due to Gatekeeper blocking an unsigned/quarantined dylib.

`understatapi` uses plain `requests` under the hood, ships as a normal
pip wheel, and hits the same underlying understat.com data.

Install:
    pip install understatapi

Run:
    python extract_understat_v2.py
"""

import logging
import time
from pathlib import Path

import pandas as pd
from understatapi import UnderstatClient

# =========================
# CONFIG
# =========================
LEAGUE = "EPL"          # understatapi's league codes: EPL, La_liga, Bundesliga, Serie_A, Ligue_1, RFPL
SEASON = "2025"         # season is the START year, e.g. "2025" = 2025/26
MAX_PLAYERS = None     # full player set now that sample sizes need to overlap with FPL/FBref
SLEEP_TIME = 0.3        # be polite between per-player requests
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_league_players(client: UnderstatClient) -> list[dict]:
    """All players who featured in the league this season, with season totals."""
    log.info("Fetching %s %s player list from Understat...", LEAGUE, SEASON)
    return client.league(league=LEAGUE).get_player_data(season=SEASON)


def get_player_match_data(client: UnderstatClient, player_id: str) -> list[dict]:
    """Per-match xG/xA/shots for a single player, across all seasons Understat has."""
    try:
        return client.player(player=player_id).get_match_data()
    except Exception as e:
        log.warning("Failed to fetch matches for player %s: %s", player_id, e)
        return []


def build_dataset(max_players: int | None = MAX_PLAYERS) -> pd.DataFrame:
    client = UnderstatClient()

    players = get_league_players(client)
    log.info("Total players found: %d", len(players))

    rows = []
    n = min(len(players), max_players) if max_players else len(players)

    for i, p in enumerate(players[:n]):
        player_id = p["id"]
        player_name = p["player_name"]
        team_name = p["team_title"]

        matches = get_player_match_data(client, player_id)

        for m in matches:
            # Only keep matches from the season we're targeting —
            # get_match_data() returns a player's full history across seasons.
            if m.get("season") != SEASON:
                continue

            rows.append({
                "understat_player_id": player_id,
                "player_name": player_name,
                "team": team_name,
                "understat_match_id": m.get("id"),
                "date": m.get("date"),
                "season": m.get("season"),
                "side": m.get("side"),          # 'h' or 'a'
                "minutes": m.get("time"),
                "goals": m.get("goals"),
                "assists": m.get("assists"),
                "shots": m.get("shots"),
                "key_passes": m.get("key_passes"),
                "xG": m.get("xG"),
                "xA": m.get("xA"),
                "npxG": m.get("npxG"),
                "npg": m.get("npg"),
                "xGChain": m.get("xGChain"),
                "xGBuildup": m.get("xGBuildup"),
                "position": m.get("position"),
            })

        log.info("[%d/%d] %s (%d matches)", i + 1, n, player_name, len(matches))
        time.sleep(SLEEP_TIME)

    return pd.DataFrame(rows)


def main():
    df = build_dataset()

    print("\nDataset created!")
    print(df.head())

    output_path = OUTPUT_DIR / "understat_player_match_stats.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print("Shape:", df.shape)

    print("\n=== Columns ===")
    print(list(df.columns))


if __name__ == "__main__":
    main()