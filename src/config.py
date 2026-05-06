FORECAST_HORIZON = 8
TEST_SIZE = 12 

DATE_COL = "Date"
TARGET_COL = "Total"
STATE_COL = "State"

MODELS = ["arima", "prophet", "xgboost", "lstm"]

MODEL_DIR = "saved_models/states/"
REGISTRY_PATH = "saved_models/registry.json"
METRICS_PATH = "saved_models/metrics.json"