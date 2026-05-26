
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout,
    Input
)

def build_lstm_gru_hybrid_model(
    input_shape
):

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(
        LSTM(
            128,
            return_sequences=True
        )
    )

    model.add(Dropout(0.3))

    model.add(GRU(64))

    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model
