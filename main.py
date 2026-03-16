# Heart Disease Prediction using Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("heart.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

# Features and target variable
X = data.drop("target", axis=1)
y = data["target"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))
