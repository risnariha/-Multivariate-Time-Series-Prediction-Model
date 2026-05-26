
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    Input
)

def build_cnn_lstm_model(input_shape):

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu'
        )
    )

    model.add(MaxPooling1D(pool_size=2))

    model.add(
        LSTM(
            64,
            return_sequences=True
        )
    )

    model.add(Dropout(0.3))

    model.add(LSTM(32))

    model.add(Dropout(0.3))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model
