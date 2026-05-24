#LSTM MODEL TRINING 

#using early stoping for mitigate overfitting

lstm_early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)


lstm_history = lstm_model.fit(

    X_train_lstm,
    y_train_lstm,

    epochs=100,

    batch_size=32,

    validation_split=0.1,

    callbacks=[
        lstm_early_stop
    ],

    verbose=1
)

#GRU MODEL TRAINING

gru_early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

#training  GRU

gru_history = gru_model.fit(

    X_train_gru,
    y_train_gru,

    epochs=50,

    batch_size=32,

    validation_split=0.1,

    callbacks=[
        gru_early_stop
    ],

    verbose=1
)


#Compared plots

plt.figure(figsize=(15, 6))

# Plot LSTM Loss
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='Train Loss')
plt.plot(lstm_history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training History')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()

# Plot GRU Loss
plt.subplot(1, 2, 2)
plt.plot(gru_history.history['loss'], label='Train Loss')
plt.plot(gru_history.history['val_loss'], label='Val Loss')
plt.title('GRU Training History')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()

plt.tight_layout()
plt.show()
