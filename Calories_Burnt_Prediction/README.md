

## Project Structure
```
📦 calories_burnt_prediction
 ┣ 📂 data
 ┃ ┣ 📄 calories.csv
 ┃ ┣ 📄 exercise.csv
 ┣ 📂 scripts
 ┃ ┣ 📄 data_preprocessing.py
 ┃ ┣ 📄 model_training.py
 ┃ ┣ 📄 prediction.py
 ┣ 📂 models
 ┣ 📄 app.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
```

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data preprocessing:
   ```bash
   python scripts/data_preprocessing.py
   ```
3. Train the model:
   ```bash
   python scripts/model_training.py
   ```
4. Make predictions:
   ```bash
   python scripts/prediction.py
   ```
5. Run the API:
   ```bash
   python app.py
   ```
   Use POST request to `/predict` with JSON input:
   ```json
   {"features": [10, 25, 180, 75]}
   ```
