import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ Load the dataset
df = pd.read_csv("TSLA_stock_data.csv", skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])

# 2️⃣ Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# 3️⃣ Sort data by date
df = df.sort_values("Date")

# 4️⃣ Normalize numerical columns using MinMaxScaler
scaler = MinMaxScaler()
df[["Close", "High", "Low", "Open", "Volume"]] = scaler.fit_transform(df[["Close", "High", "Low", "Open", "Volume"]])

# 5️⃣ Split dataset into features (X) and target variable (y)
X = df[["High", "Low", "Open", "Volume"]]
y = df["Close"]

# 6️⃣ Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 7️⃣ Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 8️⃣ Make Predictions
y_pred = model.predict(X_test)

# 9️⃣ Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 🔹 Print Performance Metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")  # Closer to 1 means better model performance

# 🔟 Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Prices", color='blue')
plt.plot(y_pred, label="Predicted Prices", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Close Price (Scaled)")
plt.title("Actual vs. Predicted Prices")
plt.legend()
plt.show()
