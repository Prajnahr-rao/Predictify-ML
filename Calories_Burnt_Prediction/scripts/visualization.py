import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
from scripts.data_preprocessing import load_and_preprocess_data

# Load the dataset
df_exercise = pd.read_csv("data/exercise.csv")
df_calories = pd.read_csv("data/calories.csv")
df = pd.merge(df_exercise, df_calories, on="User_ID")

# Load the trained model
with open("models/calories_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare test data
X_train, X_test, y_train, y_test = load_and_preprocess_data()
y_pred = model.predict(X_test)

# 1. Histogram of Calories Burnt
plt.figure(figsize=(8, 5))
sns.histplot(df['Calories'], bins=30, kde=True, color="blue")
plt.title("Distribution of Calories Burnt")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()

# 2. Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color="red")
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories Burnt")
plt.show()

# 3. Residuals Histogram (Error Analysis)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color="green")
plt.title("Residuals (Prediction Errors)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# 4. Feature vs Calories Burnt (Example: Duration)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Duration'], y=df['Calories'], color="purple")
plt.xlabel("Duration (minutes)")
plt.ylabel("Calories Burnt")
plt.title("Duration vs Calories Burnt")
plt.show()
