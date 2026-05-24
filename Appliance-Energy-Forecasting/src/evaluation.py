# Evaluate Baseline Models

def evaluate_model(y_true, y_pred, model_name):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    print(f"{model_name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print()

#EVALUATE LSTM

print("LSTM Results")

evaluate_model(
    y_test_lstm,
    lstm_pred,
    "LSTM"
)

# GRU  EVALUATION

def evaluate_model(y_true, y_pred, model_name):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(model_name)

    print("MAE:", mae)

    print("RMSE:", rmse)

    print("R2:", r2)

#EVALUATE GRU

evaluate_model(
    y_test_actual,
    gru_pred_actual,
    "GRU Actual Scale"
)
