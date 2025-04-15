from src.predict import predict_cost

# Sample input
user_input = {
    "age": 29,
    "sex": "female",
    "bmi": 26.5,
    "children": 5,
    "smoker": "yes",
    "region": "southeast"
}

predicted_charge = predict_cost(user_input)
print(f"Predicted Insurance Charge: ${predicted_charge}")
