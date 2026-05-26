
import numpy as np

def create_sequences(X, y, time_steps=24):

    Xs = []
    ys = []

    for i in range(len(X) - time_steps):

        Xs.append(X.iloc[i:(i + time_steps)].values)

        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)
