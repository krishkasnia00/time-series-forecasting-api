def create_features(df):
    df = df.copy()

    df["lag_1"] = df["Total"].shift(1)
    df["lag_7"] = df["Total"].shift(7)
    df["lag_30"] = df["Total"].shift(30)

    df["rolling_mean_7"] = df["Total"].rolling(7).mean()
    df["rolling_std_7"] = df["Total"].rolling(7).std()

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    return df.dropna()


