"""
FPL Points Predictor — Cross-Season Bonus Training Data (Bonus v2)
========================================================================
Builds a combined 24/25 + 25/26 training set for a genuine PER-PLAYER
expected-bonus regression — a different, more tractable framing than
the earlier attempt ("Bonus by pos.py" / "Bonus point pred.py"), which
tried to predict the EXACT bonus value (0/1/2/3) by ranking a player
against their real match roster and found that too hard (BPS R²=0.082,
didn't beat a naive "always 0" baseline).

This instead predicts EXPECTED bonus as a continuous/count value
directly from a player's own rolling performance — the same Poisson-
per-position approach already working for goals/assists/cards. It
doesn't need to nail the exact bonus every week, just meaningfully
separate players by real bonus-worthiness — which the current flat
POSITION-only average can't do at all (every forward gets the same
number regardless of form or role).

Uses the SAME rolling features the earlier SHAP analysis already
confirmed carry real signal for BPS: minutes, bps itself, xG, xA,
defensive contribution proxies, cards. Computed consistently across
both seasons using the same shift-first rolling-window logic as the
cards/minutes/goals-assists cross-season scripts.

Run:
    python build_bonus_training_data.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

WEIGHT_CURRENT_SEASON = 3.0
WEIGHT_HISTORICAL_SEASON = 1.0

BONUS_FEATURE_COLS = ["roll5_bps", "season_bps", "roll5_fpl_xG", "season_fpl_xG",
                       "roll5_fpl_xA", "season_fpl_xA", "roll5_CrdY",
                       "roll5_minutes", "was_home_int", "position",
                       "bonus", "player_id", "gameweek", "season"]


def add_rolling_bonus_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling/season-cumulative bps, xG, xA, cards, and
    minutes directly from the 24/25 raw per-gameweek data — same
    shift-first approach as every other cross-season script this
    project, so the current gameweek never leaks into its own feature."""
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
    add_roll_and_season("fpl_xG", "roll5_fpl_xG", "season_fpl_xG")
    add_roll_and_season("fpl_xA", "roll5_fpl_xA", "season_fpl_xA")
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

    current_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not current_path.exists():
        print(f"{current_path} not found — run the main pipeline first.")
        return
    current = pd.read_csv(current_path).copy()
    current["season"] = "2025-26"
    current["sample_weight"] = WEIGHT_CURRENT_SEASON

    # current data's own bps/minutes rolling features are already
    # precomputed under these exact names — only fpl_xG/fpl_xA need
    # checking, since current uses the richer Understat season_xG/
    # roll5_xG instead, which we deliberately do NOT alias here (same
    # honesty principle as build_goals_assists_training_data.py — don't
    # silently mix two different xG sources under one column name)
    if "fpl_xG" in current.columns and "fpl_xA" in current.columns:
        current = add_rolling_bonus_features(current)
    else:
        print("(Note: current season has no fpl_xG/fpl_xA — those two "
              "features will be sparse/absent in the combined dataset.)")

    missing = [c for c in BONUS_FEATURE_COLS if c not in current.columns]
    if missing:
        print(f"(Note: {missing} not found in current data — check column names.)")

    combined_cols = [c for c in BONUS_FEATURE_COLS if c in hist.columns and c in current.columns]
    combined = pd.concat([
        hist[combined_cols + ["sample_weight"]],
        current[combined_cols + ["sample_weight"]],
    ], ignore_index=True)

    before = len(combined)
    if "season_bps" in combined.columns:
        combined = combined.dropna(subset=["season_bps"])
    print(f"Dropped {before - len(combined)} rows with no bps history yet "
          f"(each player's first appearance in the data).")

    output_path = PROCESSED_DIR / "bonus_training_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path} ({len(combined)} rows)")
    print(f"  2024-25 rows: {(combined['season'] == '2024-25').sum()} (weight {WEIGHT_HISTORICAL_SEASON})")
    print(f"  2025-26 rows: {(combined['season'] == '2025-26').sum()} (weight {WEIGHT_CURRENT_SEASON})")
    print(f"\nBonus rate check: mean bonus = {combined['bonus'].mean():.4f} "
          f"(sanity check — should be roughly 0.2-0.35, matching the position averages found earlier)")
    print(f"Columns available for modeling: {combined_cols}")


if __name__ == "__main__":
    main()