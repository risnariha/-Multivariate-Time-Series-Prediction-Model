import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').interpolate()
    return df

def get_scalers(df, selected_features):
    scaler = MinMaxScaler()
    scaler.fit(df[selected_features])
    y_scaler = MinMaxScaler()
    y_scaler.fit(df[['Appliances']])
    return scaler, y_scaler