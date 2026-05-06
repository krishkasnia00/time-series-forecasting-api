import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


def train_all_models(df):
    df = df.copy()

    df = df.sort_values("Date")

    y = df["Total"]

    train = df.iloc[:-8]
    test = df.iloc[-8:]

    y_train = train["Total"]
    y_test = test["Total"]

    metrics = {}

    #  ARIMA
    try:
        arima_model = ARIMA(y_train, order=(1, 1, 1)).fit()
        arima_pred = arima_model.forecast(steps=8)

        rmse = np.sqrt(mean_squared_error(y_test, arima_pred))
        mae = mean_absolute_error(y_test, arima_pred)

        metrics["arima"] = {"rmse": float(rmse), "mae": float(mae)}

        arima_forecast = arima_model.forecast(steps=8).tolist()

    except Exception as e:
        print("ARIMA Error:", e)
        metrics["arima"] = {"rmse": 1e12, "mae": 1e12}
        arima_forecast = [0]*8


    # PROPHET
    try:
        prophet_df = train[["Date", "Total"]].rename(columns={"Date": "ds", "Total": "y"})

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=8, freq="W")
        forecast = model.predict(future)

        prophet_pred = forecast["yhat"].tail(8).values

        rmse = np.sqrt(mean_squared_error(y_test, prophet_pred))
        mae = mean_absolute_error(y_test, prophet_pred)

        metrics["prophet"] = {"rmse": float(rmse), "mae": float(mae)}

        prophet_forecast = forecast["yhat"].tail(8).tolist()

    except Exception as e:
        print("Prophet Error:", e)
        metrics["prophet"] = {"rmse": 1e12, "mae": 1e12}
        prophet_forecast = [0]*8


    #  XGBOOST
    try:
        features = [col for col in df.columns if col not in ["Total", "Date", "State"]]

        X_train = train[features]
        X_test = test[features]

        model = XGBRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        xgb_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        mae = mean_absolute_error(y_test, xgb_pred)

        metrics["xgboost"] = {"rmse": float(rmse), "mae": float(mae)}

        last_row = X_test.iloc[-1:]
        xgb_forecast = [float(model.predict(last_row)[0]) for _ in range(8)]

    except Exception as e:
        print("XGBoost Error:", e)
        metrics["xgboost"] = {"rmse": 1e12, "mae": 1e12}
        xgb_forecast = [0]*8


    
    best_model = min(metrics, key=lambda x: metrics[x]["rmse"])


 
    if best_model == "arima":
        final_forecast = arima_forecast
    elif best_model == "prophet":
        final_forecast = prophet_forecast
    else:
        final_forecast = xgb_forecast


    return {
        "forecast": final_forecast,  
        "best_model": best_model,
        "metrics": metrics
    }