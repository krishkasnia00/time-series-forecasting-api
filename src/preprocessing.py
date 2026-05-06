import pandas as pd
from .config import *

def load_data(path):
    df = pd.read_excel(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df

def preprocess(df):
    df = df.sort_values([STATE_COL, DATE_COL])

    all_states = df[STATE_COL].unique()
    final_df = []

    for state in all_states:
        temp = df[df[STATE_COL] == state].copy()
        temp = temp.set_index(DATE_COL).asfreq("W")

        # fill missing values
        temp[TARGET_COL] = temp[TARGET_COL].interpolate()

        temp[STATE_COL] = state
        final_df.append(temp.reset_index())

    return pd.concat(final_df)