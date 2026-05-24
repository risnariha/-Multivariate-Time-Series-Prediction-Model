# 2 FEATURE ENGINEERING

#created hour feature

df['hour'] = df['date'].dt.hour

#created time feature

df['month'] = df['date'].dt.month

df['day'] = df['date'].dt.day

df['minute'] = df['date'].dt.minute

#created lag feature

df['lag_1'] = df['Appliances'].shift(1)

df['lag_6'] = df['Appliances'].shift(6)

df['lag_12'] = df['Appliances'].shift(12)

df['lag_24'] = df['Appliances'].shift(24)

# created Rolling features


df['rolling_mean_12'] = (
    df['Appliances']
    .rolling(12)
    .mean()
)

df['rolling_mean_24'] = (
    df['Appliances']
    .rolling(24)
    .mean()
)

df['rolling_std_24'] = (
    df['Appliances']
    .rolling(24)
    .std()
)

#remove nan

df = df.dropna()

#created Interaction Features

df['temp_humidity'] = (
    df['T1'] * df['RH_1']
)

#

df = df.dropna()

#correlation analysis

correlation = (
    df.corr(numeric_only=True)['Appliances']
    .sort_values(ascending=False)
)

print(correlation)


#feature Selection

selected_features = [

    'lag_1',
    'rolling_mean_12',
    'rolling_std_24',
    'rolling_mean_24',

    'lag_6',
    'lag_12',
    'hour',
    'lights',

    'T2',
    'T6',
    'lag_24',

    'T_out',
    'temp_humidity',

    'Windspeed',
    'RH_1',
    'T3'
]

#check for 0

print(df.isna().sum().sum())

#checked infinity value

print(np.isinf(X_train_gru).sum())

print(np.isinf(y_train_gru).sum())

