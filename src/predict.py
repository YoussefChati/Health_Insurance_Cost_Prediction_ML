import joblib
import pandas as pd

# Load the model
model = joblib.load('model/insurance_model.pkl')

def predict_cost(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return round(prediction, 2)
