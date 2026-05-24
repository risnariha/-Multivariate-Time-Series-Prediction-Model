# 3 Model Development

#BASELINE MODELS

# linear Regression

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# random forest

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

{splitting codes in data_processing.py}

#  Define and Fit Linear Regression

lr = LinearRegression()
lr.fit(X_train, y_train)

#  Predict

lr_pred = lr.predict(X_test)

# random forest

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)


{evaluation baseline model codes in evaluation.py}