# Smart Home Energy Consumption Forecasting

## Overview

This project predicts smart home energy consumption using machine learning and deep learning techniques.  
It includes data preprocessing, feature engineering, sequence generation, model training, evaluation, and visualization.

The project compares several forecasting models including:

- Linear Regression
- Random Forest Regressor
- LSTM
- GRU
- Bidirectional LSTM
- CNN-LSTM Hybrid
- LSTM-GRU Hybrid

---

# Project Structure

```bash
final/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── baseline_linear_regression.pkl
│   ├── random_forest_model.pkl
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   ├── bilstm_model.h5
│   ├── cnn_lstm_model.h5
│   └── hybrid_lstm_gru_model.h5
│
├── report/
│   ├── Final_Energy_Consumption_Report.pdf
│   └── figures/
│
├── src/
│   ├── baseline_model.py
│   ├── biderectional_LSTM.py
│   ├── CNNs_LSTM HYBRID.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── GRU.py
│   ├── LSTM.py
│   ├── LSTM_GRU.py
│   ├── sequence_builder.py
│   └── train.py
    |_____ evaluate.py
│
└── README.md
```

---

# Features

## Data Processing

- Missing value handling
- Data cleaning
- Time-series formatting
- Dataset preprocessing

## Feature Engineering

The project generates advanced time-series features such as:

- Lag features
- Rolling mean
- Rolling standard deviation
- Time-based features
- Temperature interaction features

## Deep Learning Architectures

### LSTM
Captures long-term temporal dependencies in energy usage data.

### GRU
Efficient recurrent neural network for sequential forecasting.

### Bidirectional LSTM
Processes sequence data in both forward and backward directions.

### CNN-LSTM Hybrid
Combines convolutional feature extraction with temporal learning.

### LSTM-GRU Hybrid
Uses both LSTM and GRU layers for improved forecasting performance.

---

# Dataset

The dataset contains smart home sensor and appliance energy consumption measurements.

Example attributes include:

- Appliances
- Lights
- Indoor temperature
- Humidity
- Outdoor temperature
- Wind speed
- Date and time

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd final
```

## Create Virtual Environment (Optional)

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

Main libraries used:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras
- scipy

---

# Running the Project

Run the main training pipeline:

```bash
python src/train.py
```

The pipeline performs:

1. Data preprocessing
2. Feature engineering
3. Sequence generation
4. Model training
5. Evaluation
6. Saving trained models

---

# Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Predictions are compared against actual appliance energy consumption values.

---

# Output

## Trained Models

Saved in:

```bash
models/
```

## Reports and Figures

Saved in:

```bash
report/
```

---

# Future Improvements

Potential enhancements include:

- Hyperparameter optimization
- Transformer-based forecasting
- Attention mechanisms
- Real-time prediction dashboard
- Flask/FastAPI deployment
- Cloud deployment support

---

# Author

Smart Home Energy Consumption Forecasting Project

---

# License

This project is intended for educational and research purposes.