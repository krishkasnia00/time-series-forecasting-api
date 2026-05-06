from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_arima(train):
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    return model.fit(disp=False)

def predict_arima(model, steps):
    return model.forecast(steps=steps)