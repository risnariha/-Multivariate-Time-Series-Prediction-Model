import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))
from data_preprocessing import load_and_clean_data, get_scalers
from feature_engineering import create_features
from sequence_builder import create_sequences
from gru_model import build_gru_model
from evaluation import evaluate_model

DATA_PATH = '../data/raw/energydata_complete.csv'
SELECTED_FEATURES = ['lag_1', 'rolling_mean_12', 'rolling_std_24', 'rolling_mean_24', 'lag_6', 'lag_12', 'hour', 'lights', 'T2', 'T6', 'lag_24', 'T_out', 'temp_humidity', 'Windspeed', 'RH_1', 'T3']

if os.path.exists(DATA_PATH):
    df = load_and_clean_data(DATA_PATH)
    df = create_features(df)
    scaler, y_scaler = get_scalers(df, SELECTED_FEATURES)
    X_scaled = scaler.transform(df[SELECTED_FEATURES])
    y_scaled = y_scaler.transform(df[['Appliances']]).flatten()
    train_idx = int(len(df) * 0.8)
    X_train_raw, X_test_raw = X_scaled[:train_idx], X_scaled[train_idx:]
    y_train_raw, y_test_raw = y_scaled[:train_idx], y_scaled[train_idx:]
    X_train, y_train = create_sequences(pd.DataFrame(X_train_raw), pd.Series(y_train_raw))
    X_test, y_test = create_sequences(pd.DataFrame(X_test_raw), pd.Series(y_test_raw))
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    preds = model.predict(X_test)
    y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1,1))
    preds_actual = y_scaler.inverse_transform(preds)
    evaluate_model(y_test_actual, preds_actual, 'Final GRU Model')
    model.save('../models/gru_energy_model.h5')
else:
    print("Data file not found for training script.")