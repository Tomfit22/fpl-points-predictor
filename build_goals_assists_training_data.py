"""
FPL Points Predictor — Cross-Season Goals/Assists Training Data (Phase 1c)
========================================================================
Same cross-season approach as cards and minutes, applied to goals and
assists — with an honest caveat: the CURRENT 25/26 model uses richer
Understat-derived features (season_xG, season_shots, season_key_passes)
that the 24/25 archive doesn't have. What it DOES have is FPL's own
official xG/xA per gameweek (a different, generally reasonable but
less granular source) — this builds ROLLING versions of THOSE instead,
giving a scoped, honest feature set that's directly comparable across
both seasons rather than pretending to full feature parity.

Run:
    python build_goals_assists_training_data.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

WEIGHT_CURRENT_SEASON = 3.0
WEIGHT_HISTORICAL_SEASON = 1.0

# NOTE: these are FPL's own xG/xA (fpl_xG/fpl_xA), NOT the Understat
# season_xG/season_xA the current model uses — a deliberately scoped,
# honestly-different feature set, not a full replacement.
GOALS_ASSISTS_FEATURE_COLS = ["roll5_fpl_xG", "season_fpl_xG", "roll5_fpl_xA", "season_fpl_xA",
                              "was_home_int", "position", "goals_scored", "assists",
                              "player_id", "gameweek", "season"]


def add_rolling_xg_xa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling and season-cumulative versions of FPL's own per-gameweek
    xG/xA — same shift-first approach as cards/minutes to avoid
    leaking the current gameweek into its own feature."""
    df = df.sort_values(["player_id", "gameweek"]).copy()

    for raw_col, roll_col, season_col in [
        ("fpl_xG", "roll5_fpl_xG", "season_fpl_xG"),
        ("fpl_xA", "roll5_fpl_xA", "season_fpl_xA"),
    ]:
        if raw_col not in df.columns:
            continue
        df[roll_col] = (
            df.groupby("player_id")[raw_col]
            .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
        )
        df[season_col] = (
            df.groupby("player_id")[raw_col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

    return df


def main():
    hist_path = PROCESSED_DIR / "historical_2024_25_reconciled.csv"
    if not hist_path.exists():
        print(f"{hist_path} not found — run build_historical_season_data.py first.")
        return
    hist = pd.read_csv(hist_path)
    hist = hist[hist["player_id"].notna()].copy()

    if "fpl_xG" not in hist.columns or "fpl_xA" not in hist.columns:
        print("fpl_xG/fpl_xA not found in the historical data — check "
              "build_historical_season_data.py's KEEP_COLUMNS mapping.")
        return

    hist = add_rolling_xg_xa_features(hist)
    hist["sample_weight"] = WEIGHT_HISTORICAL_SEASON

    current_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not current_path.exists():
        print(f"{current_path} not found — run the main pipeline first.")
        return
    current = pd.read_csv(current_path).copy()
    current["season"] = "2025-26"
    current["sample_weight"] = WEIGHT_CURRENT_SEASON

    # current data won't have fpl_xG/fpl_xA under these exact names —
    # it has its own richer Understat-based season_xG/roll5_xG instead.
    # For the CURRENT season we compute the SAME kind of rolling feature
    # from whatever FPL-native xG/xA equivalent is available there, so
    # both sides of the combined dataset genuinely mean the same thing
    # — rather than silently mixing Understat-based and FPL-based xG
    # under one column name, which would quietly corrupt the feature.
    current_xg_col = "fpl_xG" if "fpl_xG" in current.columns else None
    current_xa_col = "fpl_xA" if "fpl_xA" in current.columns else None
    if current_xg_col is None or current_xa_col is None:
        print(f"(Note: current season data has no fpl_xG/fpl_xA columns — "
              f"only using {[c for c in ['roll5_fpl_xG','season_fpl_xG','roll5_fpl_xA','season_fpl_xA'] if c in current.columns]} "
              f"if present, or skipping this feature pair otherwise.)")
    else:
        current = add_rolling_xg_xa_features(current)

    missing = [c for c in GOALS_ASSISTS_FEATURE_COLS if c not in current.columns]
    if missing:
        print(f"(Note: {missing} not found in current data — check column names.)")

    combined_cols = [c for c in GOALS_ASSISTS_FEATURE_COLS if c in hist.columns and c in current.columns]
    if not combined_cols:
        print("No shared columns between the two seasons — cannot build a combined dataset.")
        return

    combined = pd.concat([
        hist[combined_cols + ["sample_weight"]],
        current[combined_cols + ["sample_weight"]],
    ], ignore_index=True)

    before = len(combined)
    if "season_fpl_xG" in combined.columns:
        combined = combined.dropna(subset=["season_fpl_xG"])
    print(f"Dropped {before - len(combined)} rows with no xG history yet "
          f"(each player's first appearance in the data).")

    output_path = PROCESSED_DIR / "goals_assists_training_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path} ({len(combined)} rows)")
    print(f"  2024-25 rows: {(combined['season'] == '2024-25').sum()} (weight {WEIGHT_HISTORICAL_SEASON})")
    print(f"  2025-26 rows: {(combined['season'] == '2025-26').sum()} (weight {WEIGHT_CURRENT_SEASON})")
    if "goals_scored" in combined.columns:
        print(f"\nGoals rate check: mean goals_scored = {combined['goals_scored'].mean():.4f} "
              f"(sanity check — should be a small positive number, roughly 0.05-0.15)")
    if "assists" in combined.columns:
        print(f"Assists rate check: mean assists = {combined['assists'].mean():.4f} "
              f"(similarly small, roughly 0.05-0.12)")


if __name__ == "__main__":
    main()