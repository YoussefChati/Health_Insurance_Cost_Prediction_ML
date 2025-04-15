import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv('data/insurance.csv')

# Features and label
X = df.drop('charges', axis=1)
y = df['charges']

# Categorical columns
categorical_cols = ['sex', 'smoker', 'region']

# Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'model/insurance_model.pkl')

# ✅ Make sure this line is there:
print("✅ Model trained and saved!")
