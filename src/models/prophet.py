from prophet import Prophet
import pandas as pd

def train_prophet(df):
    model = Prophet()
    model.fit(df)
    return model

def predict_prophet(model, periods):
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast["yhat"].tail(periods).values