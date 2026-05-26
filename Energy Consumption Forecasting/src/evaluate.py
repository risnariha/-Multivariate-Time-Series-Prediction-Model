import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create models directory
os.makedirs('models', exist_ok=True)

def evaluate_model(y_true, y_pred, model_name):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Results")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")