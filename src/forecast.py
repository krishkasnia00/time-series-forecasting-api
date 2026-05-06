from .models.arima import train_arima, predict_arima
from .models.prophet import train_prophet, predict_prophet
from .models.xgboost import train_xgb, predict_xgb
from .features import create_features
from .config import *

def forecast_state(state_df, best_model):

    y = state_df[TARGET_COL]

    #ARIMA
    if best_model == "arima":
        model = train_arima(y)
        forecast = predict_arima(model, 8)

    #PROPHET
    elif best_model == "prophet":
        prophet_df = state_df[[DATE_COL, TARGET_COL]].rename(
            columns={DATE_COL: "ds", TARGET_COL: "y"}
        )
        model = train_prophet(prophet_df)
        forecast = predict_prophet(model, 8)

    # XGBOOST
    elif best_model == "xgboost":
        feat_df = create_features(state_df)

        drop_cols = [TARGET_COL, STATE_COL, DATE_COL]
        if "Category" in feat_df.columns:
            drop_cols.append("Category")

        X = feat_df.drop(drop_cols, axis=1)
        y = feat_df[TARGET_COL]

        X = X.select_dtypes(include=["number"])

        model = train_xgb(X, y)

        forecast = model.predict(X.tail(8))

    else:
        forecast = []

    return list(forecast)