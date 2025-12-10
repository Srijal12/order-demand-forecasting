
# Order Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")


# Load dataset
df = pd.read_csv("orders_data.csv")
print("Dataset Preview:")
print(df.head())

# Convert date column into datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Extract time-based features
df["Day"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month

# Label encode product category
encode = LabelEncoder()
df["Product_Category"] = encode.fit_transform(df["Product_Category"])

# Select features and target
X = df.drop(["Orders", "Date"], axis=1)
y = df["Orders"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_tr, X_ts, y_tr, y_ts = train_test_split(X_scaled, y, test_size=0.2, random_state=10)
print(f"\nTraining rows: {len(X_tr)}   Test rows: {len(X_ts)}\n")

# Model selection
model_list = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=10),
    "XGBoost": XGBRegressor(random_state=10)
}

# Train and evaluate
for model_name, model in model_list.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_ts)

    mae = mean_absolute_error(y_ts, y_pred)
    rmse = np.sqrt(mean_squared_error(y_ts, y_pred))
    r2 = r2_score(y_ts, y_pred)

    print(f"Model: {model_name}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2 Score: {r2:.4f}\n")

# Final visualization using Random Forest
rf = RandomForestRegressor(random_state=10)
rf.fit(X_tr, y_tr)
pred = rf.predict(X_ts)

plt.figure(figsize=(8, 5))
plt.plot(y_ts.values, label="Actual Orders")
plt.plot(pred, label="Predicted Orders")
plt.title("Actual vs Predicted Order Count (Random Forest)")
plt.xlabel("Test Data Index")
plt.ylabel("Orders")
plt.legend()
plt.grid(True)
plt.show()


