"""
FPL Points Predictor — Optimal Squad Builder
========================================================================
Builds the genuinely BEST possible 15-player squad under real FPL
constraints, using integer programming (scipy.optimize.milp) — not a
greedy heuristic. Squad selection with a budget, position quotas, and
a max-3-per-team rule is a classic combinatorial optimization problem;
a greedy "pick the best available" approach can easily miss better
combinations that require passing on one strong pick to afford two
others. MILP finds the mathematically optimal answer given the model's
own predictions.

Decision variables (one pair per player):
  squad[i]   = 1 if player i is in the 15-man squad
  starter[i] = 1 if player i is in the starting XI (requires squad[i]=1)

Objective: maximize predicted points from the starting XI, with a small
weight on bench strength too (so bench players aren't picked at random
among otherwise-equal options — useful for autosubs).

Constraints:
  - Exactly 15 in squad: 2 GK, 5 DEF, 5 MID, 3 FWD
  - Total price <= budget
  - At most 3 players from any one real-world team
  - Exactly 11 starters, valid formation (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD)
  - A player can only start if they're in the squad

Run:
    python build_optimal_squad.py
    python build_optimal_squad.py --budget 99.5 --gameweek 3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

PROCESSED_DIR = Path("data/processed")

SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTER_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
STARTER_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
BENCH_WEIGHT = 0.1  # small — prioritizes starting XI strength, bench is a tiebreaker


def load_player_pool(gameweek: int) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "live_predictions.csv")
    df = df[df["gameweek"] == gameweek].copy()
    df = df.drop_duplicates(subset="player_id")
    df = df[df["predicted_points"].notna() & df["value"].notna()]
    df["price"] = df["value"] / 10.0  # FPL stores price as tenths (e.g. 120 -> £12.0m)
    return df.reset_index(drop=True)


def build_and_solve(df: pd.DataFrame, budget: float):
    n = len(df)
    positions = df["position"].values
    teams = df["team"].values
    prices = df["price"].values
    points = df["predicted_points"].fillna(0).values

    n_vars = 2 * n

    c = np.zeros(n_vars)
    c[:n] = -BENCH_WEIGHT * points
    c[n:] = -(1 - BENCH_WEIGHT) * points

    constraints = []

    for pos, quota in SQUAD_QUOTAS.items():
        row = np.zeros(n_vars)
        mask = (positions == pos)
        row[:n][mask] = 1
        constraints.append(LinearConstraint(row, quota, quota))

    row = np.zeros(n_vars)
    row[:n] = 1
    constraints.append(LinearConstraint(row, 15, 15))

    row = np.zeros(n_vars)
    row[:n] = prices
    constraints.append(LinearConstraint(row, 0, budget))

    for team in np.unique(teams):
        row = np.zeros(n_vars)
        mask = (teams == team)
        row[:n][mask] = 1
        constraints.append(LinearConstraint(row, 0, MAX_PER_TEAM))

    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = -1
        row[n + i] = 1
        constraints.append(LinearConstraint(row, -np.inf, 0))

    row = np.zeros(n_vars)
    row[n:] = 1
    constraints.append(LinearConstraint(row, 11, 11))

    for pos in SQUAD_QUOTAS:
        row = np.zeros(n_vars)
        mask = (positions == pos)
        row[n:][mask] = 1
        constraints.append(LinearConstraint(row, STARTER_MIN[pos], STARTER_MAX[pos]))

    bounds = Bounds(0, 1)
    integrality = np.ones(n_vars)

    result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
    return result, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=100.0, help="Total squad budget in GBP millions")
    parser.add_argument("--gameweek", type=int, default=None, help="Gameweek to optimize for (default: earliest available)")
    args = parser.parse_args()

    path = PROCESSED_DIR / "live_predictions.csv"
    if not path.exists():
        print(f"{path} not found — run build_live_predictions.py first.")
        return

    all_df = pd.read_csv(path)
    gameweek = args.gameweek if args.gameweek is not None else int(all_df["gameweek"].min())
    print(f"Optimizing squad for gameweek {gameweek}, budget GBP {args.budget}m\n")

    df = load_player_pool(gameweek)
    if df.empty:
        print(f"No players found for gameweek {gameweek} — check the gameweek number.")
        return

    print(f"Player pool: {len(df)} players")

    result, n = build_and_solve(df, args.budget)

    if not result.success:
        print(f"\nOptimization FAILED: {result.message}")
        print("This usually means the constraints are infeasible — e.g. budget too "
              "low to field a full squad, or too few players available at some position.")
        return

    x = result.x
    squad_mask = x[:n] > 0.5
    starter_mask = x[n:] > 0.5

    squad = df[squad_mask].copy()
    squad["starting"] = starter_mask[squad_mask]
    squad = squad.sort_values(["position", "starting", "predicted_points"], ascending=[True, False, False])

    total_cost = squad["price"].sum()
    starter_points = squad.loc[squad["starting"], "predicted_points"].sum()

    print(f"\n{'='*70}")
    print(f"OPTIMAL SQUAD — gameweek {gameweek}")
    print(f"{'='*70}")
    print(f"Total cost: GBP {total_cost:.1f}m / GBP {args.budget}m budget")
    print(f"Starting XI predicted points: {starter_points:.2f}\n")

    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_players = squad[squad["position"] == pos]
        print(f"--- {pos} ---")
        for _, row in pos_players.iterrows():
            marker = "  (STARTING)" if row["starting"] else "  (bench)"
            print(f"  {row['player_name']:<20} {row['team']:<16} GBP{row['price']:.1f}m  "
                  f"{row['predicted_points']:.2f} pts{marker}")
        print()

    team_counts = squad["team"].value_counts()
    print("Team distribution (max 3 allowed):")
    for team, count in team_counts.items():
        print(f"  {team}: {count}")


if __name__ == "__main__":
    main()