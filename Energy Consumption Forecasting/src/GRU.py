
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout,
    Input
)

def build_gru_model(input_shape):

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(
        GRU(
            128,
            return_sequences=True
        )
    )

    model.add(Dropout(0.3))

    model.add(GRU(64))

    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model
