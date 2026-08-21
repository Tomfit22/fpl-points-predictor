"""
FPL Points Predictor — Cards v2: Opponent/Team Features
========================================================================
Extends build_cards_training_data.py with team-level card-tendency
features — how card-prone is the OPPONENT (proxy for "dirty"/foul-
drawing playing styles), and how card-prone is the player's OWN team
(proxy for tactical aggression). The original cards model only used a
player's own individual history — this adds match-context signal that
was sitting unused in model_ready_dataset.csv all along
(opp_season_match_yellow_cards, own_season_match_yellow_cards).

Computed from the 24/25 archive by aggregating to TEAM level first
(total cards a team's players received in each match), then rolling
that team-level total the same shift-first way as every other
cross-season feature in this project — never leaking the current
match into its own feature.

Run:
    python build_cards_training_data_v2.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

WEIGHT_CURRENT_SEASON = 3.0
WEIGHT_HISTORICAL_SEASON = 1.0

CARD_FEATURE_COLS_V2 = ["season_CrdY", "roll5_CrdY", "season_CrdR", "roll5_CrdR",
                         "own_season_match_yellow_cards", "opp_season_match_yellow_cards",
                         "position", "was_home_int", "yellow_cards", "red_cards",
                         "player_id", "team", "opponent_team", "gameweek", "season"]


def add_rolling_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Player-level rolling card features — same as build_cards_training_data.py."""
    df = df.sort_values(["player_id", "gameweek"]).copy()
    for col, roll_col, season_col in [
        ("yellow_cards", "roll5_CrdY", "season_CrdY"),
        ("red_cards", "roll5_CrdR", "season_CrdR"),
    ]:
        df[roll_col] = (
            df.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
        )
        df[season_col] = (
            df.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
    return df


def add_team_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Team-level card tendency — total yellow cards a team's players
    received in each of their matches, rolled forward the same
    shift-first way, then attached both as the player's OWN team rate
    and (via a lookup on opponent_team) the OPPONENT's rate."""
    df = df.copy()

    # total cards received BY each team, per match they played
    team_match_cards = (
        df.groupby(["team", "gameweek"])["yellow_cards"].sum().reset_index()
        .rename(columns={"yellow_cards": "team_total_yellow_cards"})
    )
    team_match_cards = team_match_cards.sort_values(["team", "gameweek"])

    # season-cumulative average of that team-level total, shifted so
    # the current match never leaks into its own feature
    team_match_cards["own_season_match_yellow_cards"] = (
        team_match_cards.groupby("team")["team_total_yellow_cards"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    # attach the player's OWN team's rate directly
    df = df.merge(
        team_match_cards[["team", "gameweek", "own_season_match_yellow_cards"]],
        on=["team", "gameweek"], how="left",
    )

    # attach the OPPONENT's rate by looking up their OWN rate under
    # their own team name for the same gameweek
    opp_rates = team_match_cards[["team", "gameweek", "own_season_match_yellow_cards"]].rename(
        columns={"team": "opponent_team", "own_season_match_yellow_cards": "opp_season_match_yellow_cards"}
    )
    df = df.merge(opp_rates, on=["opponent_team", "gameweek"], how="left")

    return df


def main():
    hist_path = PROCESSED_DIR / "historical_2024_25_reconciled.csv"
    if not hist_path.exists():
        print(f"{hist_path} not found — run build_historical_season_data.py first.")
        return
    hist = pd.read_csv(hist_path)
    hist = hist[hist["player_id"].notna()].copy()

    if "opponent_team" not in hist.columns:
        print("historical_2024_25_reconciled.csv has no 'opponent_team' column — "
              "check build_historical_season_data.py's KEEP_COLUMNS mapping includes it.")
        return

    hist = add_rolling_card_features(hist)
    hist = add_team_card_features(hist)
    hist["sample_weight"] = WEIGHT_HISTORICAL_SEASON

    current_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not current_path.exists():
        print(f"{current_path} not found — run the main pipeline first.")
        return
    current = pd.read_csv(current_path).copy()
    current["season"] = "2025-26"
    current["sample_weight"] = WEIGHT_CURRENT_SEASON
    # current data already has own_season_match_yellow_cards /
    # opp_season_match_yellow_cards precomputed (confirmed earlier this
    # project) — no need to recompute, just check they're really there
    missing_team_feats = [c for c in ["own_season_match_yellow_cards", "opp_season_match_yellow_cards"]
                           if c not in current.columns]
    if missing_team_feats:
        print(f"(Note: {missing_team_feats} not found in current data — "
              f"these opponent features will be sparse/absent in the combined dataset.)")

    missing = [c for c in CARD_FEATURE_COLS_V2 if c not in current.columns]
    if missing:
        print(f"(Note: {missing} not found in current data — check column names.)")

    combined_cols = [c for c in CARD_FEATURE_COLS_V2 if c in hist.columns and c in current.columns]
    combined = pd.concat([
        hist[combined_cols + ["sample_weight"]],
        current[combined_cols + ["sample_weight"]],
    ], ignore_index=True)

    before = len(combined)
    combined = combined.dropna(subset=["season_CrdY"])
    print(f"Dropped {before - len(combined)} rows with no card history yet.")

    output_path = PROCESSED_DIR / "cards_training_combined_v2.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path} ({len(combined)} rows)")
    print(f"  2024-25 rows: {(combined['season'] == '2024-25').sum()} (weight {WEIGHT_HISTORICAL_SEASON})")
    print(f"  2025-26 rows: {(combined['season'] == '2025-26').sum()} (weight {WEIGHT_CURRENT_SEASON})")
    print(f"  Rows with opponent feature present: "
          f"{combined['opp_season_match_yellow_cards'].notna().sum() if 'opp_season_match_yellow_cards' in combined.columns else 0}")
    print(f"\nColumns available for modeling: {combined_cols}")


if __name__ == "__main__":
    main()