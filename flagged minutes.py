import pandas as pd

fpl = pd.read_csv("data/raw/fpl_clean_dataset.csv")
matching = pd.read_csv("data/processed/entity_matching_table.csv")

# total minutes played this season, per player
minutes_by_player = fpl.groupby("player_id")["minutes"].sum().reset_index()
minutes_by_player.columns = ["player_id", "total_minutes"]

merged = matching.merge(minutes_by_player, on="player_id", how="left")

flagged = merged[merged["needs_review"]]
not_flagged = merged[~merged["needs_review"]]

print("=== Minutes played: flagged vs not flagged ===")
print(f"Flagged players ({len(flagged)}): median minutes = {flagged['total_minutes'].median():.0f}, "
      f"mean = {flagged['total_minutes'].mean():.0f}")
print(f"Not flagged ({len(not_flagged)}): median minutes = {not_flagged['total_minutes'].median():.0f}, "
      f"mean = {not_flagged['total_minutes'].mean():.0f}")

print(f"\nFlagged players with 0 minutes played: {(flagged['total_minutes'] == 0).sum()} / {len(flagged)}")
print(f"Flagged players with under 90 minutes played (< 1 full match): "
      f"{(flagged['total_minutes'] < 90).sum()} / {len(flagged)}")

print("\n=== Flagged players who HAVE played significant minutes (500+) — these are the real problems ===")
real_problems = flagged[flagged["total_minutes"] >= 500].sort_values("total_minutes", ascending=False)
print(real_problems[["fpl_full_name", "fpl_team", "total_minutes", "understat_name", "understat_score",
                      "fbref_name", "fbref_score"]].to_string())