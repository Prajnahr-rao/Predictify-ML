import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    # Load datasets
    exercise_df = pd.read_csv("data/exercise.csv")
    calories_df = pd.read_csv("data/calories.csv")

    # Merge datasets on User_ID
    df = pd.merge(exercise_df, calories_df, on="User_ID")

    # Check for missing values
    df.dropna(inplace=True)

    # Encode categorical variables if any
    if 'Gender' in df.columns:
        df = pd.get_dummies(df, columns=["Gender"], drop_first=True)

    # Separate features and target
    X = df.drop(columns=["User_ID", "Calories"])
    y = df["Calories"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data preprocessing completed successfully!")
