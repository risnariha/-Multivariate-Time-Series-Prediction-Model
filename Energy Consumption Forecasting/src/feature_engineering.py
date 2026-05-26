import numpy as np

def engineer_features(df):

    # =========================
    # Time-based Features
    # =========================
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Weekend indicator
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # =========================
    # Lag Features
    # =========================
    df['lag_1'] = df['Appliances'].shift(1)
    df['lag_6'] = df['Appliances'].shift(6)
    df['lag_12'] = df['Appliances'].shift(12)
    df['lag_24'] = df['Appliances'].shift(24)

    # =========================
    # Rolling Features
    # =========================
    df['rolling_mean_6'] = (
        df['Appliances']
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    df['rolling_mean_12'] = (
        df['Appliances']
        .shift(1)
        .rolling(window=12)
        .mean()
    )

    df['rolling_std_6'] = (
        df['Appliances']
        .shift(1)
        .rolling(window=6)
        .std()
    )

    # =========================
    # Interaction Features
    # =========================
    df['temp_humidity'] = df['T1'] * df['RH_1']

    # Temperature difference
    df['indoor_outdoor_temp_diff'] = df['T1'] - df['T_out']

    # =========================
    # Drop missing values
    # =========================
    df = df.dropna()

    # =========================
    # Selected Features
    # =========================
    selected_features = [
        'lag_1',
        'lag_6',
        'lag_12',
        'lag_24',

        'rolling_mean_6',
        'rolling_mean_12',
        'rolling_std_6',

        'hour',
        'day',
        'month',
        'day_of_week',
        'is_weekend',

        'hour_sin',
        'hour_cos',

        'Lights',

        'T1',
        'T2',
        'T6',

        'RH_1',

        'T_out',
        'Windspeed',

        'temp_humidity',
        'indoor_outdoor_temp_diff'
    ]

    return df, selected_features