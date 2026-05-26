
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    Input
)

def build_bidirectional_lstm_model(
    input_shape
):

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(
        Bidirectional(
            LSTM(
                128,
                return_sequences=True
            )
        )
    )

    model.add(Dropout(0.3))

    model.add(
        Bidirectional(
            LSTM(64)
        )
    )

    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model
