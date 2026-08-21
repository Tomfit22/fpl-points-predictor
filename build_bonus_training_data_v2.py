"""
FPL Points Predictor — Bonus v3: Opponent/Team Features
========================================================================
Extends build_bonus_training_data.py with the same team-level defensive
features already proven useful elsewhere in this project for DC and
clean sheets (opp_season_shots_for, own_season_possession,
own_season_ppda) — the bonus v2 model only used a player's own rolling
performance, missing this real, previously-validated match-context
signal entirely.

Note: these specific team stats (shots/possession/PPDA) come from
Understat/FBref, NOT the 24/25 FPL-only archive — so they're only
available for the 25/26 side of the combined dataset. This is the same
honest scoping tradeoff as goals/assists' fpl_xG — real signal where
available, gracefully degrading rather than silently faking it for
24/25 rows.

Run:
    python build_bonus_training_data_v2.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

WEIGHT_CURRENT_SEASON = 3.0
WEIGHT_HISTORICAL_SEASON = 1.0

BONUS_FEATURE_COLS_V2 = ["roll5_bps", "season_bps", "roll5_CrdY", "roll5_minutes",
                          "was_home_int", "position", "bonus", "player_id",
                          "opp_season_shots_for", "own_season_possession", "own_season_ppda",
                          "gameweek", "season"]


def add_rolling_bonus_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same rolling bps/cards/minutes logic as build_bonus_training_data.py."""
    df = df.sort_values(["player_id", "gameweek"]).copy()

    def add_roll_and_season(raw_col, roll_col, season_col):
        if raw_col not in df.columns:
            return
        df[roll_col] = (
            df.groupby("player_id")[raw_col]
            .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
        )
        df[season_col] = (
            df.groupby("player_id")[raw_col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

    add_roll_and_season("bps", "roll5_bps", "season_bps")
    add_roll_and_season("yellow_cards", "roll5_CrdY", "season_CrdY")

    df["roll5_minutes"] = (
        df.groupby("player_id")["minutes"]
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    )
    return df


def main():
    hist_path = PROCESSED_DIR / "historical_2024_25_reconciled.csv"
    if not hist_path.exists():
        print(f"{hist_path} not found — run build_historical_season_data.py first.")
        return
    hist = pd.read_csv(hist_path)
    hist = hist[hist["player_id"].notna()].copy()
    hist = add_rolling_bonus_features(hist)
    hist["sample_weight"] = WEIGHT_HISTORICAL_SEASON

    # honest limitation: shots/possession/PPDA genuinely don't exist in
    # the FPL-only 24/25 archive (Understat/FBref-only stats) — leave
    # them absent for 24/25 rather than fake a value
    for col in ["opp_season_shots_for", "own_season_possession", "own_season_ppda"]:
        if col not in hist.columns:
            hist[col] = pd.NA

    current_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not current_path.exists():
        print(f"{current_path} not found — run the main pipeline first.")
        return
    current = pd.read_csv(current_path).copy()
    current["season"] = "2025-26"
    current["sample_weight"] = WEIGHT_CURRENT_SEASON

    missing_team_feats = [c for c in ["opp_season_shots_for", "own_season_possession", "own_season_ppda"]
                           if c not in current.columns]
    if missing_team_feats:
        print(f"(Note: {missing_team_feats} not found in current data — "
              f"these features will be sparse/absent in the combined dataset.)")

    missing = [c for c in BONUS_FEATURE_COLS_V2 if c not in current.columns]
    if missing:
        print(f"(Note: {missing} not found in current data — check column names.)")

    combined_cols = [c for c in BONUS_FEATURE_COLS_V2 if c in hist.columns and c in current.columns]
    combined = pd.concat([
        hist[combined_cols + ["sample_weight"]],
        current[combined_cols + ["sample_weight"]],
    ], ignore_index=True)

    before = len(combined)
    combined = combined.dropna(subset=["season_bps"])
    print(f"Dropped {before - len(combined)} rows with no bps history yet.")

    output_path = PROCESSED_DIR / "bonus_training_combined_v2.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path} ({len(combined)} rows)")
    print(f"  2024-25 rows: {(combined['season'] == '2024-25').sum()} (weight {WEIGHT_HISTORICAL_SEASON})")
    print(f"  2025-26 rows: {(combined['season'] == '2025-26').sum()} (weight {WEIGHT_CURRENT_SEASON})")
    if "opp_season_shots_for" in combined.columns:
        print(f"  Rows with opp_season_shots_for present: {combined['opp_season_shots_for'].notna().sum()}")
    print(f"\nColumns available for modeling: {combined_cols}")


if __name__ == "__main__":
    main()