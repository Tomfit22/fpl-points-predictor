"""
FPL Points Predictor — Batch Player Verification
=======================================================
Checks a list of "players who shouldn't be here" against the roster
snapshot, sorting each into one of three buckets:

  GENUINELY CURRENT  — really is on the roster right now, just an
                        unfamiliar name (very plausible for promoted
                        teams' squad depth)
  STALE/WRONG TEAM   — on the roster, but under a DIFFERENT team than
                        shown (a real bug worth investigating)
  NOT ON ROSTER AT ALL — genuinely shouldn't be appearing anywhere

Edit PLAYERS_TO_CHECK below with your list, then run:
    python check_player_list.py
"""

from pathlib import Path

import pandas as pd

import build_live_predictions as blp

PROCESSED_DIR = Path("data/processed")

# (player_name, team_shown_on_dashboard)
PLAYERS_TO_CHECK = [
    ("H. Jones", "Arsenal"), ("Malen", "Aston Villa"), ("H. Traorè", "Brentford"),
    ("Cox", "Brentford"), ("Veltman", "Chelsea"), ("Cashin", "Chelsea"),
    ("Milner", "Chelsea"), ("Estève", "Coventry City"), ("Walker", "Coventry City"),
    ("Hartman", "Coventry City"), ("Lucas Pires", "Coventry City"), ("Weiss", "Coventry City"),
    ("Yalcouye", "Coventry City"), ("Moran", "Coventry City"), ("Knight", "Coventry City"),
    ("Sarmiento", "Coventry City"), ("Delcroix", "Coventry City"), ("Dodgson", "Coventry City"),
    ("Roberts", "Coventry City"), ("Green", "Coventry City"), ("Hladký", "Coventry City"),
    ("Sonne", "Coventry City"), ("Sambo", "Coventry City"), ("Jordan", "Coventry City"),
    ("Hannibal", "Crystal Palace"), ("Benson", "Crystal Palace"), ("Adewumi", "Crystal Palace"),
    ("Édouard", "Hull City"), ("Ebiowei", "Hull City"), ("Ozoh", "Hull City"),
    ("Rodney", "Hull City"), ("Umeh", "Hull City"), ("Dragusin", "Hull City"),
    ("Agbinone", "Hull City"), ("Ahamada", "Hull City"), ("Coleman", "Hull City"),
    ("Tyrer", "Hull City"), ("Marsh", "Hull City"), ("Luís Hemir", "Hull City"),
    ("Welch", "Hull City"), ("Gana", "Ipswich Town"), ("Heath", "Ipswich Town"),
    ("Bates", "Ipswich Town"), ("Ba", "Ipswich Town"), ("Abdullahi", "Ipswich Town"),
    ("Metcalfe", "Ipswich Town"), ("Benda", "Ipswich Town"), ("Y. Chermiti", "Ipswich Town"),
    ("Amissah", "Ipswich Town"), ("Harris", "Leeds"), ("M. Salah", "Liverpool"),
    ("Greenwood", "Liverpool"), ("Fitzgerald", "Newcastle"), ("Triantis", "Newcastle"),
    ("Krafth", "Nott'm Forest"), ("Matete", "Nott'm Forest"),
]


def main():
    roster = pd.read_csv(PROCESSED_DIR / "current_roster_snapshot.csv")

    genuinely_current, wrong_team, not_on_roster = [], [], []

    for name, shown_team in PLAYERS_TO_CHECK:
        norm_query = blp.normalize_name(name)
        matches = roster[roster["player_name"].apply(blp.normalize_name) == norm_query]

        if matches.empty:
            not_on_roster.append((name, shown_team))
            continue

        real_team = matches.iloc[0]["team"]
        if blp.normalize_name(real_team) == blp.normalize_name(shown_team):
            genuinely_current.append((name, shown_team))
        else:
            wrong_team.append((name, shown_team, real_team))

    print(f"=== GENUINELY CURRENT ({len(genuinely_current)}) — really on this roster right now ===")
    for name, team in genuinely_current:
        print(f"  {name} ({team})")

    print(f"\n=== WRONG TEAM SHOWN ({len(wrong_team)}) — on the roster, but under a DIFFERENT team — real bug ===")
    for name, shown, real in wrong_team:
        print(f"  {name}: dashboard shows '{shown}', roster actually shows '{real}'")

    print(f"\n=== NOT ON ROSTER AT ALL ({len(not_on_roster)}) — shouldn't be appearing anywhere — real bug ===")
    for name, team in not_on_roster:
        print(f"  {name} ({team})")


if __name__ == "__main__":
    main()