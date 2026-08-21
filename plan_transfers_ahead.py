"""
FPL Points Predictor — Multi-Gameweek Transfer Planner
========================================================================
Plans transfers ACROSS several upcoming gameweeks, not just optimizing
one week in isolation — a genuinely different problem from
build_optimal_squad.py (which stays untouched, single-gameweek only,
as requested).

Approach: rolling-horizon greedy. At each gameweek, every player (owned
or available) gets scored by their SUM of predicted points over a
look-ahead window (default 3 gameweeks) — capturing "is this player
good for the near future", not just the current week. This naturally
finds fixture-swing value: a cheap player with a strong short run
scores highly now; once that run ends and rolls out of the window in
later iterations, a better alternative will outscore them, flagging a
sell.

At each gameweek, the single best available transfer (if any) gets
evaluated: free transfers are taken if they improve the look-ahead
score at all; anything beyond the free transfer(s) available needs to
clear a real bar — the score improvement must be worth more than the
-4 point hit.

HONEST LIMITATIONS, stated plainly rather than glossed over:
  - This is a GREEDY rolling-horizon heuristic, not a proven globally
    optimal multi-week plan — finding the true joint optimum across N
    weeks simultaneously is a vastly harder problem. This gives a
    genuinely reasonable, practical plan, not a guaranteed-best one.
  - Sell price is assumed equal to buy price (no price-change modeling
    — this project doesn't predict transfer-market price rises/falls).
  - Free transfers accumulate at 1 per week, banking up to a maximum
    of 5 (confirmed current FPL rule).

Run:
    python plan_transfers_ahead.py --team-file my_team.txt --weeks 5
    python plan_transfers_ahead.py --team-file my_team.txt --weeks 5 --bank 0.5
"""

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

LOOKAHEAD_WINDOW = 3       # gameweeks used to score "near-future value"
FREE_TRANSFERS_PER_WEEK = 1
MAX_BANKED_TRANSFERS = 5   # confirmed current FPL rule allows banking up to 5
TRANSFER_HIT_COST = 4
MIN_MEANINGFUL_GAIN = 1.0  # even FREE transfers need to clear this bar — without
                            # it, tiny noise-level gains (e.g. +0.05 pts) get taken
                            # just because they're free, which can flip-flop the
                            # same swap back and forth as the look-ahead window
                            # shifts by one gameweek each iteration


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def load_team_from_file(path: Path, pool: pd.DataFrame) -> pd.DataFrame:
    raw_names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    pool = pool.copy()
    pool["_norm"] = pool["player_name"].apply(normalize_name)

    matched_rows, unmatched = [], []
    for name in raw_names:
        norm = normalize_name(name)
        candidates = pool[pool["_norm"] == norm]
        if len(candidates) == 0:
            unmatched.append(name)
        else:
            matched_rows.append(candidates.iloc[0])

    if unmatched:
        print(f"*** {len(unmatched)} name(s) couldn't be matched: {unmatched} ***\n")
    if not matched_rows:
        return pd.DataFrame()
    return pd.DataFrame(matched_rows).drop(columns=["_norm"])


def score_lookahead(all_df: pd.DataFrame, from_gw: int, window: int) -> pd.Series:
    """Sums predicted_points over [from_gw, from_gw+window) per player_id."""
    window_df = all_df[(all_df["gameweek"] >= from_gw) & (all_df["gameweek"] < from_gw + window)]
    return window_df.groupby("player_id")["predicted_points"].sum()


def find_best_transfer(squad_ids: set, scores: pd.Series, pool_at_gw: pd.DataFrame, bank: float):
    """Finds the single best (sell, buy) pair — same position, affordable
    within sale price + bank, not already owned, genuinely higher
    look-ahead score. Returns None if nothing improves the squad."""
    best = None
    squad_rows = pool_at_gw[pool_at_gw["player_id"].isin(squad_ids)]

    for _, sell_row in squad_rows.iterrows():
        sell_price = sell_row["value"] / 10.0
        sell_score = scores.get(sell_row["player_id"], 0)
        budget = sell_price + bank

        candidates = pool_at_gw[
            (pool_at_gw["position"] == sell_row["position"]) &
            (~pool_at_gw["player_id"].isin(squad_ids)) &
            (pool_at_gw["value"] / 10.0 <= budget)
        ].copy()
        if candidates.empty:
            continue

        candidates["lookahead_score"] = candidates["player_id"].map(scores).fillna(0)
        candidates = candidates.sort_values("lookahead_score", ascending=False)
        top_candidate = candidates.iloc[0]

        gain = top_candidate["lookahead_score"] - sell_score
        if gain <= 0:
            continue
        if best is None or gain > best["gain"]:
            best = {
                "sell_id": sell_row["player_id"], "sell_name": sell_row["player_name"],
                "sell_team": sell_row["team"], "sell_price": sell_price,
                "buy_id": top_candidate["player_id"], "buy_name": top_candidate["player_name"],
                "buy_team": top_candidate["team"], "buy_price": top_candidate["value"] / 10.0,
                "gain": gain,
            }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-file", type=str, required=True)
    parser.add_argument("--weeks", type=int, default=5, help="How many gameweeks to plan ahead")
    parser.add_argument("--bank", type=float, default=0.0, help="Money in the bank, GBP millions")
    parser.add_argument("--start-gameweek", type=int, default=None)
    args = parser.parse_args()

    team_file = Path(args.team_file)
    if not team_file.exists():
        print(f"{team_file} not found.")
        return

    path = PROCESSED_DIR / "live_predictions.csv"
    if not path.exists():
        print(f"{path} not found - run build_live_predictions.py first.")
        return

    all_df = pd.read_csv(path)
    start_gw = args.start_gameweek if args.start_gameweek is not None else int(all_df["gameweek"].min())

    first_pool = all_df[all_df["gameweek"] == start_gw].drop_duplicates(subset="player_id").copy()
    my_team = load_team_from_file(team_file, first_pool)
    if my_team.empty:
        print("No players matched - nothing to plan.")
        return

    squad_ids = set(my_team["player_id"])
    bank = args.bank
    banked_transfers = FREE_TRANSFERS_PER_WEEK

    print("=" * 70)
    print(f"TRANSFER PLAN - gameweeks {start_gw} to {start_gw + args.weeks - 1}")
    print("=" * 70)
    print(f"(Rolling-horizon greedy plan, {LOOKAHEAD_WINDOW}-gameweek look-ahead window. "
          f"See the script's docstring for honest limitations - this is a reasonable, "
          f"practical plan, not a proven globally optimal one.)\n")

    for week_offset in range(args.weeks):
        gw = start_gw + week_offset
        pool_at_gw = all_df[all_df["gameweek"] == gw].drop_duplicates(subset="player_id")
        if pool_at_gw.empty:
            print(f"GW{gw}: no data available - stopping plan here.")
            break

        scores = score_lookahead(all_df, gw, LOOKAHEAD_WINDOW)

        transfer = find_best_transfer(squad_ids, scores, pool_at_gw, bank)

        print(f"--- GW{gw} (free transfers available: {banked_transfers}) ---")

        if transfer is None:
            print("  No improving transfer found - hold your squad.\n")
        else:
            is_free = banked_transfers > 0
            worth_a_hit = transfer["gain"] > TRANSFER_HIT_COST
            meaningful = transfer["gain"] >= MIN_MEANINGFUL_GAIN

            if meaningful and (is_free or worth_a_hit):
                cost_note = "FREE transfer" if is_free else f"costs -{TRANSFER_HIT_COST} pts (hit taken because the gain justifies it)"
                print(f"  TRANSFER: OUT {transfer['sell_name']} ({transfer['sell_team']}, "
                      f"GBP{transfer['sell_price']:.1f}m) -> IN {transfer['buy_name']} "
                      f"({transfer['buy_team']}, GBP{transfer['buy_price']:.1f}m)")
                print(f"  Look-ahead score gain over next {LOOKAHEAD_WINDOW} gameweeks: +{transfer['gain']:.2f} pts")
                print(f"  {cost_note}")

                squad_ids.discard(transfer["sell_id"])
                squad_ids.add(transfer["buy_id"])
                bank = bank + transfer["sell_price"] - transfer["buy_price"]
                if is_free:
                    banked_transfers -= 1
            elif not meaningful:
                print(f"  A transfer exists (OUT {transfer['sell_name']} -> IN {transfer['buy_name']}) "
                      f"but the gain (+{transfer['gain']:.2f} pts) is too small to be worth using a "
                      f"transfer on, even a free one - holding instead.")
            else:
                print(f"  A transfer exists (+{transfer['gain']:.2f} pts: OUT {transfer['sell_name']} "
                      f"-> IN {transfer['buy_name']}) but doesn't clear the -{TRANSFER_HIT_COST} hit "
                      f"threshold and no free transfer is banked - holding instead.")
            print()

        banked_transfers = min(banked_transfers + FREE_TRANSFERS_PER_WEEK, MAX_BANKED_TRANSFERS)

    print(f"Final bank: GBP{bank:.1f}m")


if __name__ == "__main__":
    main()
