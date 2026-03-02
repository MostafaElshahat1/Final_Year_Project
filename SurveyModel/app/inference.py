import joblib
import pandas as pd
import os

# Get the path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'mental_health_model.pkl')

class ModelInference:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, data_df: pd.DataFrame):
        prediction = self.model.predict(data_df)[0]
        probability = self.model.predict_proba(data_df)[0][1]
        return prediction, probability