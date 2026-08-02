"""
FPL Points Predictor — Minutes Model Coefficient Inspection
=================================================================
Henderson's OWN features are at the ceiling (roll5_starts=1.0,
roll5_minutes=90.0) going into the prediction, yet the model still
caps out around 91-98%. This fits the actual minutes model on real
data and prints its coefficients directly — showing exactly what's
capping the prediction, rather than speculating.

Run:
    python check_minutes_model_coefficients.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
MINUTES_FEATURES = ["roll5_starts", "consecutive_starts", "days_since_last_game", "roll5_minutes"]


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[df["roll5_minutes"].notna()]

    for threshold, label in [(1, "any_minutes"), (60, "sixty_plus")]:
        print(f"\n{'=' * 60}")
        print(f"MODEL: {label} (minutes >= {threshold})")
        print(f"{'=' * 60}")

        features = [f for f in MINUTES_FEATURES if f in df.columns]
        X = sm.add_constant(df[features].fillna(0))
        y = (df["minutes"] >= threshold).astype(int)
        model = sm.Logit(y, X).fit(disp=0)

        print(model.summary2().tables[1].to_string())

        print(f"\nBase rate in training data (% of rows with minutes >= {threshold}): {y.mean():.1%}")

        # Henderson's actual best-case feature vector, maxed out
        best_case = {"roll5_starts": 1.0, "roll5_minutes": 90.0, "consecutive_starts": 7, "days_since_last_game": 7.0}
        x_vec = pd.DataFrame([{**{"const": 1.0}, **{f: best_case.get(f, 0) for f in features}}])
        x_vec = x_vec[["const"] + features]
        pred = model.predict(x_vec).iloc[0]
        print(f"\nPredicted probability for a player with THESE EXACT best-case features "
              f"{best_case}: {pred:.1%}")

        # what's the theoretical ceiling if EVERY feature were pushed to its
        # most favorable extreme observed in the real data?
        extreme = {}
        for f in features:
            coef_sign = model.params[f]
            extreme[f] = df[f].max() if coef_sign > 0 else df[f].min()
        x_extreme = pd.DataFrame([{**{"const": 1.0}, **extreme}])
        x_extreme = x_extreme[["const"] + features]
        pred_extreme = model.predict(x_extreme).iloc[0]
        print(f"Theoretical ceiling if every feature were pushed to its most favorable "
              f"observed extreme {extreme}: {pred_extreme:.1%}")


if __name__ == "__main__":
    main()