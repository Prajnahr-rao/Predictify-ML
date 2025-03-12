import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    df = pd.read_csv("data/processed_data.csv")
    
    # Define features and target variable
    X = df.drop(columns=["Calories"])
    y = df["Calories"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data preprocessing complete. Training and testing sets are ready.")
