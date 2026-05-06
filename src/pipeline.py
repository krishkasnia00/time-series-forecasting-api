import os
import json
import pandas as pd

from src.preprocessing import load_data
from src.features import create_features
from src.train import train_all_models


def run_pipeline(path):
    print(" Running Pipeline...")

    # LOAD DATA 
    df = load_data(path)

    forecasts = {}
    metrics = {}

    states = df["State"].unique()

    #  LOOP STATE-WISE
    for state in states:
        print(f" Processing: {state}")

        state_df = df[df["State"] == state].copy()

        # Feature Engineering
        state_df = create_features(state_df)

        # Train models
        result = train_all_models(state_df)

        forecasts[state] = result["forecast"]
        metrics[state] = {
            "best_model": result["best_model"],
            "metrics": result["metrics"]
        }

    # SAVE OUTPUT 
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/forecasts.json", "w") as f:
        json.dump(forecasts, f, indent=4)

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(" Pipeline Completed!")

    return forecasts, metrics