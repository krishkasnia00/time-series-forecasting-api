from xgboost import XGBRegressor

def train_xgb(X, y):
    model = XGBRegressor(n_estimators=200)
    model.fit(X, y)
    return model

def predict_xgb(model, X):
    return model.predict(X)