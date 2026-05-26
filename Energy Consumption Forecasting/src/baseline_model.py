
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

def train_baseline_models(X_train, X_test, y_train, y_test):

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, pred)
        )

        print(f"{name} MAE: {mae:.4f}")
        print(f"{name} RMSE: {rmse:.4f}")
