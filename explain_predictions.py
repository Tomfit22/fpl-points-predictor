"""
FPL Points Predictor — Prediction Explainer
=================================================
Turns the raw point-contribution breakdown already sitting in
live_predictions.csv into plain-English explanations. Not a new model
— just formatting numbers we already computed into sentences a person
would actually read. Three related outputs from the same underlying
data:

  1. EXPLAIN a prediction — why did this player get this score?
  2. SCOUTING REPORT — pros/cons framing for a specific player
  3. EXPLAIN AN ERROR — after the fact, compare predicted vs actual
     and explain the gap (needs pipeline_predictions.csv, which has
     both predicted AND actual for held-out gameweeks)

Deliberately does NOT try to explain things this project hasn't
earned the right to explain — e.g. it won't claim high confidence
just because a number looks precise. Flags real uncertainty (low
minutes probability, sparse defensive-contribution history) using
signals we already compute, the same way the analysis scripts do.

Run:
    python explain_predictions.py --player "Player Name"
    python explain_predictions.py --top 5
    python explain_predictions.py --scouting "Player Name"
    python explain_predictions.py --error "Player Name" --gameweek 34
"""

import argparse
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

COMPONENT_LABELS = {
    "pts_appearance": "playing time",
    "pts_goals": "goal threat",
    "pts_assists": "creativity/assists",
    "pts_dc": "defensive contribution",
    "pts_clean_sheet": "clean sheet chance",
    "pts_saves": "saves",
    "pts_pen_saves": "penalty saves",
    "pts_cards": "card risk",
    "pts_bonus": "bonus (soft estimate)",
}

# below this many points, a component isn't worth mentioning in prose —
# keeps explanations focused on what actually matters for this player
MENTION_THRESHOLD = 0.15


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "live_predictions.csv")
    return df


def find_player(df: pd.DataFrame, name: str) -> pd.Series:
    """Exact normalized name match first — avoids the substring-collision
    problem found earlier this project (a loose .str.contains("Son")
    also matching Anderson/Wilson/Robertson/etc, since "son" is a
    substring of all of them). Falls back to substring search only if
    no exact match exists, and raises clearly on genuine ambiguity
    rather than silently picking "the first" — this tool is used
    interactively, so asking the person to be more specific is better
    than a silent guess."""
    from build_live_predictions import normalize_name

    norm_query = normalize_name(name)
    exact = df[df["player_name"].apply(normalize_name) == norm_query]

    if exact.empty:
        substring = df[df["player_name"].str.contains(name, case=False, na=False)]
        if substring.empty:
            raise ValueError(f"No player found matching '{name}'")
        exact = substring  # fall through to the same distinct-player logic below

    # a single real player can legitimately have MANY rows (one per
    # gameweek in live_predictions.csv) — that's not ambiguity, it's
    # normal. Only genuinely different (team, position) combinations
    # under the same name count as a real collision (e.g. two actual
    # different real "Onana"s).
    id_cols = ["player_id"] if "player_id" in exact.columns else ["team", "position"]
    distinct_players = exact.drop_duplicates(subset=id_cols)

    if len(distinct_players) == 1:
        if "gameweek" in exact.columns:
            exact = exact.sort_values("gameweek")
        return exact.iloc[0]

    print(f"Multiple different players matching '{name}':")
    for _, row in distinct_players.iterrows():
        print(f"  {row['player_name']} ({row['team']}, {row['position']})")
    raise ValueError(f"Ambiguous — please search using team as well, e.g. filter your "
                      f"own copy of the dataframe, or check the printed list above.")


def get_uncertainty_flags(row: pd.Series) -> list:
    """Real uncertainty signals we already compute elsewhere in this
    project — not invented for this script. A number can be precisely
    calculated and still deserve a caveat."""
    flags = []
    if row.get("pred_p_any_minutes", 1) < 0.5:
        flags.append(f"Low chance of featuring at all ({row['pred_p_any_minutes']:.0%}) — "
                      f"this prediction is much less reliable than it looks.")
    elif row.get("pred_p_60plus", 1) < 0.5 and row.get("pred_p_any_minutes", 1) >= 0.5:
        flags.append(f"Likely to feature but real risk of an early substitution "
                      f"(only {row['pred_p_60plus']:.0%} chance of 60+ minutes).")
    if row["position"] == "FWD" and row.get("pts_dc", 0) > 0:
        flags.append("Defensive contribution prediction for forwards is based on very few "
                      "historical events league-wide — treat as directional, not precise.")
    return flags


def explain_prediction(row: pd.Series):
    print(f"\n{'=' * 60}")
    print(f"{row['player_name']} ({row['team']}, {row['position']}) — Gameweek {row['gameweek']}")
    print(f"{'=' * 60}")
    print(f"Predicted: {row['predicted_points']:.1f} points\n")

    contributions = [(COMPONENT_LABELS[c], row[c]) for c in COMPONENT_LABELS if c in row.index]
    contributions = [(label, val) for label, val in contributions if abs(val) >= MENTION_THRESHOLD]
    contributions.sort(key=lambda x: -abs(x[1]))

    if contributions:
        print("Main contributors:")
        for label, val in contributions:
            sign = "+" if val >= 0 else ""
            print(f"  {sign}{val:.2f} pts — {label}")
    else:
        print("No single component stands out — a broadly low-confidence prediction overall.")

    flags = get_uncertainty_flags(row)
    if flags:
        print("\nWorth knowing:")
        for f in flags:
            print(f"  - {f}")


def scouting_report(row: pd.Series):
    print(f"\n{'=' * 60}")
    print(f"SCOUTING REPORT: {row['player_name']} ({row['team']}, {row['position']})")
    print(f"{'=' * 60}")
    print(f"Predicted: {row['predicted_points']:.1f} points | "
          f"Price: £{row['value']/10:.1f}m | Owned: {row['selected']/1e6:.2f}M managers\n")

    pros, cons = [], []
    if row.get("pts_goals", 0) > 0.5:
        pros.append(f"Real goal threat ({row['pred_goals']:.2f} expected goals)")
    if row.get("pts_assists", 0) > 0.4:
        pros.append(f"Creating chances ({row['pred_assists']:.2f} expected assists)")
    if row.get("pred_p_clean_sheet", 0) > 0.4 and row["position"] in ("GK", "DEF", "MID"):
        pros.append(f"Good clean sheet chance ({row['pred_p_clean_sheet']:.0%})")
    if row.get("pred_p_dc_hit", 0) > 0.3:
        pros.append(f"Strong defensive contribution threat ({row['pred_p_dc_hit']:.0%} chance of hitting threshold)")
    if row.get("pred_p_60plus", 0) > 0.85:
        pros.append("Nailed-on for 60+ minutes")

    if row.get("pred_p_any_minutes", 1) < 0.6:
        cons.append(f"Real doubt over involvement at all ({row['pred_p_any_minutes']:.0%})")
    elif row.get("pred_p_60plus", 1) < 0.6:
        cons.append(f"Substitution risk — only {row['pred_p_60plus']:.0%} chance of 60+ minutes")
    if row.get("pts_cards", 0) < -0.1:
        cons.append("Elevated card risk")
    if row["position"] == "FWD" and row.get("pred_p_clean_sheet", 0) > 0:
        cons.append("No clean sheet points available at this position")

    print("Pros:" if pros else "Pros: none standing out this week")
    for p in pros:
        print(f"  + {p}")
    print("\nCons:" if cons else "\nCons: none standing out this week")
    for c in cons:
        print(f"  - {c}")


def explain_error(row: pd.Series):
    predicted = row["predicted_points"]
    actual = row["actual_points"]
    error = predicted - actual

    print(f"\n{'=' * 60}")
    print(f"ERROR EXPLANATION: {row['player_name']} — Gameweek {row['gameweek']}")
    print(f"{'=' * 60}")
    print(f"Predicted: {predicted:.1f} | Actual: {actual:.1f} | Error: {error:+.1f}\n")

    if abs(error) < 1.5:
        print("This is within the model's normal error range (MAE is typically ~1.0-1.3 "
              "points) — not really a 'miss' worth explaining, just ordinary variance.")
        return

    if actual > predicted:
        print("Under-predicted. Likely explanation: this project's models predict EXPECTED "
              "value from pre-match form — they cannot see one-off events (a deflection, a "
              "refereeing decision, a moment of individual quality) that drove an unusually "
              "good match. This is expected, irreducible uncertainty, not necessarily a bug.")
    else:
        print("Over-predicted. Check first whether pred_p_any_minutes/pred_p_60plus were high "
              "but the player was actually subbed early or rested — that's the single most "
              "common cause of a large over-prediction in this pipeline's known bias analysis.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=str, help="Explain a specific player's prediction")
    parser.add_argument("--scouting", type=str, help="Scouting report for a specific player")
    parser.add_argument("--error", type=str, help="Explain a prediction error (needs pipeline_predictions.csv)")
    parser.add_argument("--top", type=int, help="Explain the top N predicted scorers")
    args = parser.parse_args()

    if args.top:
        df = load_predictions().sort_values("predicted_points", ascending=False).head(args.top)
        for _, row in df.iterrows():
            explain_prediction(row)
        return

    if args.player:
        df = load_predictions()
        explain_prediction(find_player(df, args.player))
        return

    if args.scouting:
        df = load_predictions()
        scouting_report(find_player(df, args.scouting))
        return

    if args.error:
        df = pd.read_csv(PROCESSED_DIR / "pipeline_predictions.csv")
        if "actual_points" not in df.columns:
            print("pipeline_predictions.csv doesn't have actual_points — run build_prediction_pipeline.py first.")
            return
        row = find_player(df, args.error)
        explain_error(row)
        return

    print("No action specified. Use --player, --scouting, --error, or --top. See --help.")


if __name__ == "__main__":
    main()