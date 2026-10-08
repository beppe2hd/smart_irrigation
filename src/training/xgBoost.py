import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------
# 1. Example multivariate time series
# ---------------------------------------------------
print("ciao")
np.random.seed(42)
n = 500
t = np.arange(n)

df = pd.DataFrame({
    "soil_moisture": 0.3 + 0.05*np.sin(t/20) - 0.002*t/10 + np.random.normal(0, 0.01, n),
    "temperature_2m": 20 + 5*np.sin(t/15) + np.random.normal(0, 0.8, n),
    "rain": np.random.choice([0, 0, 0, 0.5, 1.0, 2.0], size=n, p=[0.7, 0.1, 0.08, 0.06, 0.04, 0.02]),
    "relative_humidity_2m": 60 + 10*np.cos(t/18) + np.random.normal(0, 2, n)
})

# optional: make target depend a bit on the exogenous variables
df["soil_moisture"] = (
    0.85 * df["soil_moisture"]
    + 0.015 * df["rain"]
    - 0.002 * (df["temperature_2m"] - 20)
    + 0.001 * (df["relative_humidity_2m"] - 60)
)

# ---------------------------------------------------
# 2. Build supervised dataset
# ---------------------------------------------------
def create_multivariate_multistep_dataset(data, target_col, input_cols, n_in, n_out):
    X, y = [], []
    values = data[input_cols].values
    target = data[target_col].values

    for i in range(len(data) - n_in - n_out + 1):
        # past multivariate window: shape (n_in, n_features)
        x_window = values[i:i+n_in]

        # flatten to 1D because XGBoost expects tabular input
        X.append(x_window.flatten())

        # future target only: shape (n_out,)
        y.append(target[i+n_in:i+n_in+n_out])

    return np.array(X), np.array(y)

input_cols = ["soil_moisture", "temperature_2m", "rain", "relative_humidity_2m"]
target_col = "soil_moisture"

n_in = 24   # past window
n_out = 6   # forecast horizon

X, y = create_multivariate_multistep_dataset(
    data=df,
    target_col=target_col,
    input_cols=input_cols,
    n_in=n_in,
    n_out=n_out
)

print("X shape:", X.shape)  # (samples, 24 * 4)
print("y shape:", y.shape)  # (samples, 6)

# ---------------------------------------------------
# 3. Chronological train/test split
# ---------------------------------------------------
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------------------------------
# 4. Train XGBoost multi-output model
# ---------------------------------------------------
base_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# ---------------------------------------------------
# 5. Predict
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# error per forecast horizon
for h in range(n_out):
    mae_h = mean_absolute_error(y_test[:, h], y_pred[:, h])
    print(f"Step t+{h+1} MAE: {mae_h:.4f}")

# ---------------------------------------------------
# 7. Forecast from the last available window
# ---------------------------------------------------
last_window = df[input_cols].values[-n_in:].flatten().reshape(1, -1)
future_forecast = model.predict(last_window)[0]

print("Next 6 predicted soil_moisture values:")
print(future_forecast)