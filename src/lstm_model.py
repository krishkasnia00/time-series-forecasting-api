import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_lstm(series):
    series = series.values.reshape(-1,1)

    X, y = [], []
    for i in range(30, len(series)):
        X.append(series[i-30:i])
        y.append(series[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1],1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, verbose=0)

    return model