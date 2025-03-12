import pickle
import numpy as np
from scripts.data_preprocessing import load_and_preprocess_data

def predict_calories(sample_data):
    # Load trained model
    with open("models/calories_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Ensure sample_data has the correct number of features (7 features expected)
    if len(sample_data) != 7:
        raise ValueError(f"Expected 7 features, but got {len(sample_data)}")
    
    # Convert to numpy array and reshape for prediction
    prediction = model.predict(np.array(sample_data).reshape(1, -1))
    return prediction[0]

if __name__ == "__main__":
    # Example sample input (modify according to dataset features)
    sample_input = [25, 1, 70, 175, 30, 8, 120]  # Replace with actual feature values
    predicted_calories = predict_calories(sample_input)
    print(f"Predicted Calories Burnt: {predicted_calories}")
