import numpy as np
import joblib

# Assuming X_train, X_test, y_train, y_test are created during training
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Assuming "model" is your trained model (e.g., an SVR or XGBoost model)
joblib.dump(model, 'svr_model.pkl')

print("All files saved successfully!")
