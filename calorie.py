import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Load the data
calories = pd.read_csv('calories.csv')
exercise_data = pd.read_csv('exercise.csv')

# Combine the two DataFrames
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# Convert categorical data to numerical
calories_data['Gender'] = calories_data['Gender'].map({'male': 0, 'female': 1})

# Separate features and target
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Save the model
with open('calorie_predictor_model.pkl', 'wb') as file:
    pickle.dump(model, file)
