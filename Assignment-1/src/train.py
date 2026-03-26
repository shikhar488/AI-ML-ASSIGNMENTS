import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

print("Started Model Training")

# Loaded the dataset
df = pd.read_csv("data/kc_house_data.csv")

# Droppinng unnecessary columns
df = df.drop(columns=["id", "date"])

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Model Training Completed")