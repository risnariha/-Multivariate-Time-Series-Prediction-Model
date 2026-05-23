# Appliance Energy Forecasting Using Deep Learning

## Project Overview

This project focuses on forecasting household appliance energy consumption using deep learning models. The objective is to analyze historical energy usage patterns and environmental sensor data to predict future appliance energy consumption accurately.

The project implements and compares two recurrent neural network architectures:

- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The workflow includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Sequence generation
- Deep learning model training
- Model evaluation and optimization

---

# Dataset

Dataset used:
- Appliance Energy Prediction Dataset

The dataset contains:
- Appliance energy consumption
- Indoor temperature readings
- Humidity sensor values
- Weather conditions
- Time-based information

Target variable:
```text
Appliances
```

---

# Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

---

# Project Structure

```text
Appliance-Energy-Forecasting/
│
├── data/
│   ├── raw/
│   │   └── energydata_complete.csv
│   │
│   └── processed/
│       └── processed_energy_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── sequence_builder.py
│   ├── lstm_model.py
│   ├── gru_model.py
│   ├── evaluation.py
│   └── train.py
│
├── models/
│   ├── lstm_energy_model.h5
│   └── gru_energy_model.h5
│
├── reports/
│   └── report.pdf
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# Exploratory Data Analysis (EDA)

The dataset was analyzed to identify:
- consumption trends
- temporal patterns
- feature relationships
- correlations between variables

Key findings:
- Historical appliance usage had the strongest predictive power
- Lag features significantly improved forecasting
- Some environmental features had weak correlations and were removed

---

# Data Preprocessing

The following preprocessing steps were applied:

- Missing value handling
- Feature scaling using MinMaxScaler
- Train-test split
- Sequence generation for deep learning models

Missing values generated from lag and rolling operations were removed using:

```python
df = df.dropna()
```

---

# Feature Engineering

Several engineered features were created to improve model performance.

## Lag Features

```text
lag_1
lag_6
lag_12
lag_24
```

## Rolling Features

```text
rolling_mean_12
rolling_mean_24
rolling_std_24
```

## Time Features

```text
hour
lights
```

Feature selection was performed using correlation analysis to remove weak and noisy features.

---

# Deep Learning Models

## LSTM Model

Architecture:

```text
LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(1)
```

---

## GRU Model

Architecture:

```text
GRU(128) → Dropout → GRU(64) → Dropout → Dense(1)
```

---

# Model Training

Training configuration:

```text
Epochs: 100
Batch Size: 64
Optimizer: Adam
Loss Function: Huber Loss
Time Steps: 24
```

EarlyStopping was used to reduce overfitting.

---

# Results

## LSTM Results

| Metric | Value |
|---|---|
| MAE | 35.75 |
| RMSE | 65.22 |
| R² | 0.448 |

---

## GRU Results

| Metric | Value |
|---|---|
| MAE | 30.51 |
| RMSE | 63.74 |
| R² | 0.473 |

---

## Optimized GRU Results

| Metric | Value |
|---|---|
| MAE | 0.027 |
| RMSE | 0.056 |
| R² | 0.528 |

The GRU model achieved better forecasting performance than the LSTM model.

---

# Challenges Faced

## NaN Loss Problem

Feature engineering operations created missing values, causing:

```text
loss = nan
val_loss = nan
```

Solution:
```python
df = df.dropna()
```

---

## Over-Complex Architecture

Larger GRU architectures reduced performance due to overfitting.

Solution:
- simplified architecture
- reduced noisy features
- optimized feature selection

---

# Future Improvements

Potential future enhancements include:

- Bidirectional GRU
- Attention mechanisms
- Transformer models
- Hyperparameter optimization
- Ensemble forecasting methods

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Appliance-Energy-Forecasting.git
```

Move into the project directory:

```bash
cd Appliance-Energy-Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# How to Run

Run the training pipeline:

```bash
python src/train.py
```

---

# Model Files

Trained models are stored inside:

```text
models/
```

Saved model formats:
- `.h5`

---

# Report

The complete project report is available in:

```text
reports/report.pdf
```

---

# Author

## Risna Riha

Undergraduate Student  
AI / Machine Learning / Full Stack Development Enthusiast

---

# License

This project is developed for educational and assessment purposes.