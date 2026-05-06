from fastapi import FastAPI, HTTPException
import json
import os

app = FastAPI()

METRICS_PATH = "outputs/metrics.json"
FORECAST_PATH = "outputs/forecasts.json"


# LOAD FUNCTION 
def load_json(path):
    if not os.path.exists(path):
        raise Exception(f" {path} not found. Run pipeline first!")

    with open(path, "r") as f:
        content = f.read().strip()
        if not content:
            raise Exception(f" {path} is empty. Run pipeline again!")

        return json.loads(content)


#LOAD FILES 
metrics = load_json(METRICS_PATH)
forecasts = load_json(FORECAST_PATH)


#ROUTES
@app.get("/")
def home():
    return {"message": "Forecast API Running "}


@app.get("/states")
def get_states():
    return {"states": list(metrics.keys())}


@app.get("/predict/{state}")
def predict(state: str):
    state = state.strip()

    if state not in metrics:
        raise HTTPException(status_code=404, detail="State not found")

    return {
        "state": state,
        "best_model": metrics[state]["best_model"],  
        "forecast": forecasts[state],                 
        "metrics": metrics[state]["metrics"]
    }