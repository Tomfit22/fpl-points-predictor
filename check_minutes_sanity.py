"""
FPL Points Predictor — Minutes Prediction Sanity Check
============================================================
Saka showed only 29% chance of 60+ minutes — surprisingly low for one
of Arsenal's most nailed-on starters. This checks whether that's
widespread (a real pattern worth investigating) or specific to Saka
(more likely a one-off worth a closer individual look).

Run:
    python check_minutes_sanity.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

# a small set of players who should be about as close to "guaranteed
# starter, plays 60+ minutes" as anyone in the league, if healthy
KNOWN_NAILED_ON = ["Haaland", "Salah", "Saka", "B.Fernandes", "Palmer", "Son"]


def main():
    df = pd.read_csv(PROCESSED_DIR / "live_predictions.csv")

    # use each player's EARLIEST available gameweek row, consistent with
    # what explain_predictions.py itself returns by default
    earliest = df.sort_values("gameweek").groupby("player_name", as_index=False).first()

    print("=== Known nailed-on starters — pred_p_60plus ===")
    for name in KNOWN_NAILED_ON:
        matches = earliest[earliest["player_name"].str.contains(name, case=False, na=False)]
        for _, row in matches.iterrows():
            print(f"  {row['player_name']} ({row['team']}): {row['pred_p_60plus']:.0%} "
                  f"(any minutes: {row['pred_p_any_minutes']:.0%})")

    print(f"\n=== Overall distribution of pred_p_60plus across all {len(earliest)} players ===")
    print(earliest["pred_p_60plus"].describe().to_string())

    print("\n=== Same, restricted to high-value players (top price quartile — should skew high) ===")
    price_threshold = earliest["value"].quantile(0.75)
    high_value = earliest[earliest["value"] >= price_threshold]
    print(f"(price >= £{price_threshold/10:.1f}m, {len(high_value)} players)")
    print(high_value["pred_p_60plus"].describe().to_string())


if __name__ == "__main__":
    main()