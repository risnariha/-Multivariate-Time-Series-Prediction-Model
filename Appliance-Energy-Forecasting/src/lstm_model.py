#Sequence creation function
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# CREATE LSTM SEQUENCES

TIME_STEPS = 24

X_train_lstm, y_train_lstm = create_sequences(
    X_train,
    y_train,
    TIME_STEPS
)

X_test_lstm, y_test_lstm = create_sequences(
    X_test,
    y_test,
    TIME_STEPS
)

#convert to float

X_train_lstm = X_train_lstm.astype(np.float32)
y_train_lstm = y_train_lstm.astype(np.float32)

X_test_lstm = X_test_lstm.astype(np.float32)
y_test_lstm = y_test_lstm.astype(np.float32)


#LSTM building

lstm_model = Sequential()

lstm_model.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(
            X_train_lstm.shape[1],
            X_train_lstm.shape[2]
        )
    )
)

lstm_model.add(Dropout(0.3))

lstm_model.add(
    LSTM(64)
)

lstm_model.add(Dropout(0.3))

lstm_model.add(
    Dense(
        16,
        activation='relu'
    )
)

lstm_model.add(Dense(1))

# Generate sequences using the function defined earlier

TIME_STEPS = 24
X_train_lstm, y_train_lstm = create_sequences(pd.DataFrame(X_train_p), pd.Series(y_train_p), TIME_STEPS)
X_test_lstm, y_test_lstm = create_sequences(pd.DataFrame(X_test_p), pd.Series(y_test_p), TIME_STEPS)

#Build LSTM Model using modern Input layer again 

lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])

# compiling the model

lstm_optimizer = Adam(
    learning_rate=0.0005
)

lstm_model.compile(
    optimizer=lstm_optimizer,
    loss='mse',
    metrics=['mae']
)



#LSTM PREDICTION

lstm_pred = lstm_model.predict(
    X_test_lstm
)