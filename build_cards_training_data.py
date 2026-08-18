"""
FPL Points Predictor — Cross-Season Card Training Data (Phase 1)
========================================================================
Builds a combined 24/25 + 25/26 training set for a REAL card model,
replacing the naive roll5_CrdY passthrough currently used in
build_live_predictions.py (lines ~586-587) — there was never an actual
fitted model for cards before this.

The two source datasets don't share the same engineered features
(model_ready_dataset.csv has rich rolling/opponent-adjusted stats from
build_features.py; the 24/25 historical archive only has raw per-
gameweek counts) — so this computes a SCOPED, matching set of rolling
card features directly from the 24/25 raw data, using the same
rolling-window logic, rather than pulling in the full feature pipeline.

Per explicit instruction, 25/26 data is weighted more heavily than
24/25 in the 'sample_weight' column — recent data should dominate the
fitted model, with the older season mainly helping STABILIZE the fit,
especially for red cards where events are extremely sparse.

Run:
    python build_cards_training_data.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

# 25/26 rows count this many times more than 24/25 rows during fitting —
# recent data dominates, older data mainly adds stability for sparse
# events (especially red cards).
WEIGHT_CURRENT_SEASON = 3.0
WEIGHT_HISTORICAL_SEASON = 1.0

CARD_FEATURE_COLS = ["season_CrdY", "roll5_CrdY", "season_CrdR", "roll5_CrdR",
                      "position", "was_home_int", "yellow_cards", "red_cards",
                      "player_id", "gameweek", "season"]


def add_rolling_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes season-cumulative and rolling-5 card rates directly from
    raw per-gameweek yellow_cards/red_cards — same rolling-window
    LOGIC as build_features.py uses for 25/26, applied here to the
    24/25 historical data which doesn't have these precomputed."""
    df = df.sort_values(["player_id", "gameweek"]).copy()

    for col, roll_col, season_col in [
        ("yellow_cards", "roll5_CrdY", "season_CrdY"),
        ("red_cards", "roll5_CrdR", "season_CrdR"),
    ]:
        # rolling average of the PREVIOUS 5 games (shift first so the
        # current gameweek's own card isn't included in its own feature
        # — leaking the target into the feature would be a real bug)
        df[roll_col] = (
            df.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
        )
        # season-cumulative average of all PREVIOUS games this season
        df[season_col] = (
            df.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

    return df


def main():
    # --- 24/25 historical data ---
    hist_path = PROCESSED_DIR / "historical_2024_25_reconciled.csv"
    if not hist_path.exists():
        print(f"{hist_path} not found — run build_historical_season_data.py first.")
        return
    hist = pd.read_csv(hist_path)
    hist = hist[hist["player_id"].notna()].copy()  # only rows that reconciled to a current player
    hist = add_rolling_card_features(hist)
    hist["sample_weight"] = WEIGHT_HISTORICAL_SEASON

    # --- 25/26 current data (already has these features precomputed) ---
    current_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not current_path.exists():
        print(f"{current_path} not found — run the main pipeline first.")
        return
    current = pd.read_csv(current_path)
    current = current.copy()
    current["season"] = "2025-26"
    current["sample_weight"] = WEIGHT_CURRENT_SEASON

    missing = [c for c in CARD_FEATURE_COLS if c not in current.columns]
    if missing:
        print(f"(Note: {missing} not found in current data — check column names.)")

    combined_cols = [c for c in CARD_FEATURE_COLS if c in hist.columns and c in current.columns]
    combined = pd.concat([
        hist[combined_cols + ["sample_weight"]],
        current[combined_cols + ["sample_weight"]],
    ], ignore_index=True)

    # drop rows with no rolling history yet (first game each — genuinely
    # no prior data to compute a rate from) rather than silently
    # filling with 0, which would look like "never gets cards" instead
    # of "unknown yet"
    before = len(combined)
    combined = combined.dropna(subset=["season_CrdY"])
    print(f"Dropped {before - len(combined)} rows with no card history yet "
          f"(each player's first appearance in the data).")

    output_path = PROCESSED_DIR / "cards_training_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path} ({len(combined)} rows)")
    print(f"  2024-25 rows: {(combined['season'] == '2024-25').sum()} (weight {WEIGHT_HISTORICAL_SEASON})")
    print(f"  2025-26 rows: {(combined['season'] == '2025-26').sum()} (weight {WEIGHT_CURRENT_SEASON})")
    print(f"\nYellow card rate check: mean yellow_cards = {combined['yellow_cards'].mean():.4f} "
          f"(sanity check — should be a small positive number, roughly 0.05-0.15)")
    print(f"Red card rate check: mean red_cards = {combined['red_cards'].mean():.4f} "
          f"(should be very small, likely under 0.01)")


if __name__ == "__main__":
    main()