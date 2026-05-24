# 1 Data Understanding and Preprocessing

df.head()

df.shape
df.info()
df.describe()

df['date'] = pd.to_datetime(df['date'])

df = df.sort_values('date')

import pandas as pddf = df.sort_values('date')

#Check Missing Values

df.isnull().sum()

#Handle Missing Values

df = df.interpolate()

# Outlier Detection

plt.figure(figsize=(8,4))

sns.boxplot(x=df['Appliances'])

plt.title("Appliance Energy Consumption Outliers")

plt.show()

#HEATMAP (correlation)

plt.figure(figsize=(18,12))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

#plotted Hourly Energy Consumption

plt.figure(figsize=(12,5))

sns.boxplot(
    x='hour',
    y='Appliances',
    data=df
)

plt.title("Hourly Energy Consumption")

plt.show()

#double checked

print(X_scaled.head())

print(X_scaled.shape)


# 1. Prepare data and split for baseline models

X_baseline = df[selected_features]
y_baseline = df['Appliances']

# Split the data (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(
    X_baseline, 
    y_baseline, 
    test_size=0.2, 
    shuffle=False
)

#LSTM DATA SCALING, SPILLITING

scaler_lstm = MinMaxScaler()
y_scaler_lstm = MinMaxScaler()

X_scaled_lstm = scaler_lstm.fit_transform(df[selected_features])
y_scaled_lstm = y_scaler_lstm.fit_transform(df[['Appliances']]).flatten()

train_split = int(len(df) * 0.8)
X_train_p, X_test_p = X_scaled_lstm[:train_split], X_scaled_lstm[train_split:]
y_train_p, y_test_p = y_scaled_lstm[:train_split], y_scaled_lstm[train_split:]


#data scaling for GRU

scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(df[selected_features])

y_scaler = MinMaxScaler()

y_scaled = y_scaler.fit_transform(
    df[['Appliances']].values
)

X_scaled = pd.DataFrame(
    scaled_features, 
    columns=selected_features
)

y_scaled = pd.Series(
    y_scaled.flatten()
)

#covert

X_scaled = pd.DataFrame(
    X_scaled,
    columns=selected_features
)

y_scaled = pd.Series(
    y_scaled.values.flatten()
)


# GRU model TRAIN TEST SPLITTING 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df,
    y_scaled_series,
    test_size=0.2,
    shuffle=False
)

print(f"X shape: {X_scaled_df.shape}, y shape: {y_scaled_series.shape}")

#GRU

print(y_scaled.min())

print(y_scaled.max())