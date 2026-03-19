# Assisted by ChatGPT

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/kc_house_data.csv")

# Drop unnecessary columns
df = df.drop(columns=["id", "date"])

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train Test Split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load("model/model.pkl")

# Prediction
preds = model.predict(X_test)

# Calculate RMSE manually (version safe)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

# Calculate R2 Score
r2 = r2_score(y_test, preds)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, preds)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.savefig("plots/prediction.png")

print("Evaluation Completed Successfully")