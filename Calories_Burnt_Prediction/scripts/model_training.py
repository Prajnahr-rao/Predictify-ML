import pickle
from sklearn.linear_model import LinearRegression
from scripts.data_preprocessing import load_and_preprocess_data
import os

def train_model():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Save trained model
    with open("models/calories_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model training completed and saved successfully!")

if __name__ == "__main__":
    train_model()
