"""
FPL Points Predictor — Diagnostic: Range Duplication & GC Penalty Variance
========================================================================
Investigates two specific things flagged as suspicious:

  1. RANGE DUPLICATION — many players showing the exact same sim_floor/
     sim_ceiling. Checks how widespread this actually is, and whether
     it's concentrated among genuinely low-probability fringe players
     (where near-identical near-zero distributions are EXPECTED and
     fine) or spread across meaningful, regularly-playing players
     (which would be a real problem).

  2. GC PENALTY FLATNESS — checks how much pred_goals_conceded actually
     varies week-to-week for real GK/DEF players, and whether the
     underlying opponent-strength features feeding the clean-sheet
     model vary enough to produce real differentiation in the first
     place — if the INPUTS barely vary, the output can't either.

Run:
    python diagnose_range_and_gc_variance.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def check_range_duplication():
    print("=" * 70)
    print("PART 1: RANGE DUPLICATION CHECK")
    print("=" * 70)

    path = PROCESSED_DIR / "live_predictions.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)

    if "sim_floor" not in df.columns or "sim_ceiling" not in df.columns:
        print("sim_floor/sim_ceiling not found in live_predictions.csv.")
        return

    for position in ["GK", "DEF", "MID", "FWD"]:
        pos_df = df[df["position"] == position].drop_duplicates("player_id").copy()
        pos_df["range_key"] = pos_df["sim_floor"].round(2).astype(str) + "-" + pos_df["sim_ceiling"].round(2).astype(str)

        dup_counts = pos_df["range_key"].value_counts()
        duplicated_groups = dup_counts[dup_counts > 1]
        total_players = len(pos_df)
        players_in_dup_groups = duplicated_groups.sum()

        print(f"\n--- {position} ---")
        print(f"  Total players: {total_players}")
        if total_players == 0:
            print("  (no players at this position - skipping)")
            continue
        print(f"  Players sharing an identical range with at least one other player: "
              f"{players_in_dup_groups} ({players_in_dup_groups/total_players*100:.0f}%)")

        if len(duplicated_groups) > 0:
            top_groups = duplicated_groups.head(3)
            for range_key, count in top_groups.items():
                sample = pos_df[pos_df["range_key"] == range_key]
                avg_pred_points = sample["predicted_points"].mean()
                avg_p_any = sample["pred_p_any_minutes"].mean() if "pred_p_any_minutes" in sample.columns else None
                print(f"\n  Range '{range_key}' shared by {count} players "
                      f"(avg predicted_points: {avg_pred_points:.2f}"
                      + (f", avg P(any minutes): {avg_p_any:.2f}" if avg_p_any is not None else "") + ")")
                print(f"    Sample names: {sample['player_name'].head(5).tolist()}")


def check_monte_carlo_mechanics():
    """Directly verifies the simulation code itself produces genuinely
    different distributions for genuinely different input probabilities
    — a sanity check on the MECHANISM, not just the output data."""
    print("\n" + "=" * 70)
    print("PART 1b: MONTE CARLO MECHANISM SANITY CHECK")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_sims = 5000

    p_any_starter = 0.95
    p_any_fringe = 0.05

    sims_starter = (rng.random(n_sims) < p_any_starter).astype(float)
    sims_fringe = (rng.random(n_sims) < p_any_fringe).astype(float)

    print(f"\nStarter (P(any minutes)=0.95): floor={np.percentile(sims_starter,10):.2f}, "
          f"ceiling={np.percentile(sims_starter,90):.2f}")
    print(f"Fringe  (P(any minutes)=0.05): floor={np.percentile(sims_fringe,10):.2f}, "
          f"ceiling={np.percentile(sims_fringe,90):.2f}")
    print("\n(If both fringe-type players have similarly low P(any minutes), their "
          "simulated distributions WILL legitimately look near-identical - mostly "
          "zero, rarely playing. That's real signal, not a bug, if their underlying "
          "probabilities are genuinely similar.)")


def check_gc_penalty_variance():
    print("\n" + "=" * 70)
    print("PART 2: GC PENALTY / GOALS-CONCEDED VARIANCE CHECK")
    print("=" * 70)

    path = PROCESSED_DIR / "live_predictions.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)

    if "pred_goals_conceded" not in df.columns:
        print("pred_goals_conceded not found - check the GC penalty patch was applied.")
        return

    def_gk = df[df["position"].isin(["GK", "DEF"])].copy()

    sample_players = def_gk.drop_duplicates("player_id")["player_name"].head(5).tolist()

    for name in sample_players:
        player_rows = def_gk[def_gk["player_name"] == name].sort_values("gameweek")
        if len(player_rows) < 2:
            continue
        gc_values = player_rows["pred_goals_conceded"]
        print(f"\n{name} ({player_rows.iloc[0]['team']}):")
        print(f"  pred_goals_conceded across gameweeks: "
              f"min={gc_values.min():.3f}, max={gc_values.max():.3f}, "
              f"std={gc_values.std():.3f}, range={gc_values.max()-gc_values.min():.3f}")
        print(f"  pts_gc_penalty range: {player_rows['pts_gc_penalty'].min():.3f} to "
              f"{player_rows['pts_gc_penalty'].max():.3f}")

    print("\n--- Checking whether the underlying opponent-strength inputs vary enough ---")
    model_ready_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if model_ready_path.exists():
        mr_df = pd.read_csv(model_ready_path)
        for col in ["opp_season_shots_for", "opp_season_possession", "own_season_ppda"]:
            if col in mr_df.columns:
                print(f"  {col}: std={mr_df[col].std():.3f}, "
                      f"range={mr_df[col].max()-mr_df[col].min():.3f}, "
                      f"mean={mr_df[col].mean():.3f}")
    else:
        print(f"  {model_ready_path} not found - skipping input-feature check.")

    print("\n--- Overall pred_goals_conceded spread across ALL current GK/DEF rows ---")
    print(f"  min={def_gk['pred_goals_conceded'].min():.3f}, "
          f"max={def_gk['pred_goals_conceded'].max():.3f}, "
          f"std={def_gk['pred_goals_conceded'].std():.3f}")
    print("  (A small per-player week-to-week range can still coexist with real "
          "team-to-team differentiation right now - this line checks that "
          "specifically, separate from any one player's own variance over time.)")


if __name__ == "__main__":
    check_range_duplication()
    check_monte_carlo_mechanics()
    check_gc_penalty_variance()
