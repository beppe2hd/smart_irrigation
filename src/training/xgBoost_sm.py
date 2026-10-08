import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



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

input_cols = ['s_b','s_w','LAI','irr','temperature_2m','rain']
target_col = "s_w"

n_in = 72   # past window
n_out = 48   # forecast horizon


# ---------------------------------------------------
# 1. Load dataset multivariate time series
# ---------------------------------------------------
test = "./data/csv_files2/fieldv2_f17"
df_test = pd.read_csv(test)

X_test, y_test = create_multivariate_multistep_dataset(
    data=df_test,
    target_col=target_col,
    input_cols=input_cols,
    n_in=n_in,
    n_out=n_out
)

print("X Test shape:", X_test.shape)  
print("y Test shape:", y_test.shape)  

training = ["./data/csv_files2/fieldv2_f1"]#,
            #   "./data/csv_files2/fieldv2_f2",
            #   "./data/csv_files2/fieldv2_f3",
            #   "./data/csv_files2/fieldv2_f4",
            #   "./data/csv_files2/fieldv2_f5",
            #   "./data/csv_files2/fieldv2_f6",
            #   "./data/csv_files2/fieldv2_f7",
            #   "./data/csv_files2/fieldv2_f8",
            #   "./data/csv_files2/fieldv2_f9",
            #   "./data/csv_files2/fieldv2_f10",
            #   "./data/csv_files2/fieldv2_f11",
            #   "./data/csv_files2/fieldv2_f12",
            #   "./data/csv_files2/fieldv2_f13",
            #   "./data/csv_files2/fieldv2_f14",
            #   "./data/csv_files2/fieldv2_f15",
            #   "./data/csv_files2/fieldv2_f16",
            #   "./data/csv_files2/fieldv2_f18"]

X_train = []
y_train = [] 

for plot in training:

    df_training = pd.read_csv(plot)

    X_train_c, y_train_c = create_multivariate_multistep_dataset(
        data=df_training,
        target_col=target_col,
        input_cols=input_cols,
        n_in=n_in,
        n_out=n_out
    )


    X_train.append(X_train_c)
    y_train.append(y_train_c)

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)


print("X Train shape:", X_train.shape)  
print("y Train shape:", y_train.shape)
 


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

abs_error = abs(y_test, y_pred)
meanError = abs_error.mean(dim=0)
meanError = meanError.mean()
