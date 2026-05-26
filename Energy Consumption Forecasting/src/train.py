import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

def main():
    FILE_PATH = '/content/energy_data_set.csv'
    df = preprocess_data(FILE_PATH)
    df, selected_features = engineer_features(df)
    X = df[selected_features]
    y = df['Appliances']
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    train_baseline_models(X_train, X_test, y_train, y_test)

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    y_scaled = pd.Series(y_scaled)

    X_train_scaled, X_test_scaled = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
    y_train_scaled, y_test_scaled = y_scaled.iloc[:split_index], y_scaled.iloc[split_index:]

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled)

    models = {'LSTM': build_lstm_model, 'GRU': build_gru_model, 'BiLSTM': build_bidirectional_lstm_model, 'CNN-LSTM': build_cnn_lstm_model, 'LSTM-GRU': build_lstm_gru_hybrid_model}

    last_y_actual, last_pred_actual = None, None

    for name, builder in models.items():
        print(f"Training {name}")
        model = builder((X_train_seq.shape[1], X_train_seq.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train_seq, y_train_seq, validation_split=0.1, epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

        pred = model.predict(X_test_seq)
        last_pred_actual = scaler_y.inverse_transform(pred)
        last_y_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))

        evaluate_model(last_y_actual, last_pred_actual, name)
        model.save(f'models/{name}.keras')

    print("Training Completed")
    return last_y_actual, last_pred_actual

if __name__ == '__main__':
    y_actual, pred_actual = main()