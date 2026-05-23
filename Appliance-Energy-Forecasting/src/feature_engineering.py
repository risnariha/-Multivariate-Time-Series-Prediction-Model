import pandas as pd

def create_features(df):
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['lag_1'] = df['Appliances'].shift(1)
    df['lag_6'] = df['Appliances'].shift(6)
    df['lag_12'] = df['Appliances'].shift(12)
    df['lag_24'] = df['Appliances'].shift(24)
    df['rolling_mean_12'] = df['Appliances'].rolling(12).mean()
    df['rolling_mean_24'] = df['Appliances'].rolling(24).mean()
    df['rolling_std_24'] = df['Appliances'].rolling(24).std()
    df['temp_humidity'] = df['T1'] * df['RH_1']
    return df.dropna()