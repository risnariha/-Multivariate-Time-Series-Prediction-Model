#GRU MODEL

TIME_STEPS = 24

#GRU MODEL building

gru_model = Sequential()

# Explicit Input layer to resolve UserWarning
gru_model.add(Input(shape=(X_train_gru.shape[1], X_train_gru.shape[2])))

gru_model.add(
    GRU(
        128,
        return_sequences=True
    )
)

gru_model.add(Dropout(0.2))

gru_model.add(
    GRU(64)
)

gru_model.add(Dropout(0.2))

gru_model.add(Dense(1))

#check nan value

print(np.isnan(X_train_gru).sum())

print(np.isnan(y_train_gru).sum())

#compile GRU

gru_optimizer = Adam(
    learning_rate=0.0003
)

gru_model.compile(
    optimizer=gru_optimizer,
    loss='mse',
    metrics=['mae']
)


#GRU PREDICTION

gru_pred_actual = y_scaler.inverse_transform(
    gru_pred
)

y_test_actual = y_scaler.inverse_transform(
    y_test_gru.reshape(-1,1)
)