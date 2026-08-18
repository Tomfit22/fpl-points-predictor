"""
FPL Points Predictor — Cross-Season Minutes Training Data (Phase 1b)
========================================================================
Same approach as build_cards_training_data.py, applied to minutes.
This one is a genuinely strong candidate for cross-season enhancement:
every feature the current minutes model uses (roll5_minutes,
roll5_starts, consecutive_starts, days_since_last_game) is fully
computable from the 24/25 archive's raw per-gameweek data — no
Understat/FBref-only features needed here, unlike goals/assists/CS.

Run:
    python build_minutes_training_data.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

WEIGHT_CURRENT_SEASON = 3.0
WEIGHT_HISTORICAL_SEASON = 1.0

MINUTES_FEATURE_COLS = ["roll5_minutes", "roll5_starts", "consecutive_starts",
                         "days_since_last_game", "minutes", "starts",
                         "position", "player_id", "gameweek", "season"]


def add_rolling_minutes_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes roll5_minutes, roll5_starts, consecutive_starts, and
    days_since_last_game directly from raw per-gameweek minutes/starts
    — same rolling-window logic as build_features.py uses for 25/26,
    same shift-first approach as the cards script to avoid leaking the
    current gameweek into its own feature."""
    df = df.sort_values(["player_id", "gameweek"]).copy()

    df["roll5_minutes"] = (
        df.groupby("player_id")["minutes"]
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    )
    df["roll5_starts"] = (
        df.groupby("player_id")["starts"]
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    )

    # consecutive_starts: how many of the immediately preceding games
    # (walking backward) were starts, stopping at the first non-start
    def consecutive_prior_starts(starts_series):
        prior = starts_series.shift(1).fillna(0).astype(int).tolist()
        result = []
        streak = 0
        for i in range(len(prior)):
            if prior[i] == 1:
                streak += 1
            else:
                streak = 0
            result.append(streak)
        return pd.Series(result, index=starts_series.index)

    df["consecutive_starts"] = (
        df.groupby("player_id")["starts"].transform(consecutive_prior_starts)
    )

    # days_since_last_game: not reliably computable without real match
    # dates for every gameweek in this archive (kickoff_time wasn't
    # kept in the historical extraction) — use gameweek GAP as a proxy
    # (assumes ~7 days between consecutive gameweeks, ~0 signal for
    # blank/double weeks, which are rare enough not to matter much here)
    df["days_since_last_game"] = (
        df.groupby("player_id")["gameweek"].transform(lambda s: s.diff().fillna(7) * 7)
    )

    return df


def main():
    hist_path = PROCESSED_DIR / "historical_2024_25_reconciled.csv"
    if not hist_path.exists():
        print(f"{hist_path} not found — run build_historical_season_data.py first.")
        return
    hist = pd.read_csv(hist_path)
    hist = hist[hist["player_id"].notna()].copy()
    hist = add_rolling_minutes_features(hist)
    hist["sample_weight"] = WEIGHT_HISTORICAL_SEASON

    current_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not current_path.exists():
        print(f"{current_path} not found — run the main pipeline first.")
        return
    current = pd.read_csv(current_path).copy()
    current["season"] = "2025-26"
    current["sample_weight"] = WEIGHT_CURRENT_SEASON

    missing = [c for c in MINUTES_FEATURE_COLS if c not in current.columns]
    if missing:
        print(f"(Note: {missing} not found in current data — check column names.)")

    combined_cols = [c for c in MINUTES_FEATURE_COLS if c in hist.columns and c in current.columns]
    combined = pd.concat([
        hist[combined_cols + ["sample_weight"]],
        current[combined_cols + ["sample_weight"]],
    ], ignore_index=True)

    before = len(combined)
    combined = combined.dropna(subset=["roll5_minutes"])
    print(f"Dropped {before - len(combined)} rows with no minutes history yet "
          f"(each player's first appearance in the data).")

    output_path = PROCESSED_DIR / "minutes_training_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path} ({len(combined)} rows)")
    print(f"  2024-25 rows: {(combined['season'] == '2024-25').sum()} (weight {WEIGHT_HISTORICAL_SEASON})")
    print(f"  2025-26 rows: {(combined['season'] == '2025-26').sum()} (weight {WEIGHT_CURRENT_SEASON})")
    print(f"\nAny-minutes rate check: mean (minutes>0) = {(combined['minutes'] > 0).mean():.4f} "
          f"(sanity check — typically 0.6-0.75 across a full squad, since not everyone plays every week)")
    print(f"60+ minutes rate check: mean (minutes>=60) = {(combined['minutes'] >= 60).mean():.4f} "
          f"(typically somewhat lower than the any-minutes rate)")


if __name__ == "__main__":
    main()