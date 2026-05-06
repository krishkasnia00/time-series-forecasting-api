import json
from src.config import REGISTRY_PATH

def get_best_model(state):
    with open(REGISTRY_PATH) as f:
        data = json.load(f)

    return data.get(state, None)