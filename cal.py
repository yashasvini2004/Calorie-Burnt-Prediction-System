from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('calorie_predictor_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('c.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from request data
    gender = data['Gender']
    age = int(data['Age'])
    body_temp = float(data['BodyTemp'])
    heart_rate = int(data['HeartRate'])
    weight = float(data['Weight'])
    height = int(data['Height'])
    duration = int(data['Duration'])

    # Convert gender to numerical
    gender = 0 if gender.lower() == 'male' else 1

    # Create feature array
    features = np.array([[gender, age, body_temp, heart_rate, weight, height, duration]])

    # Predict calories burned
    prediction = model.predict(features)

    # Convert prediction to standard Python float
    prediction_float = float(prediction[0])

    # Return the prediction as JSON
    return jsonify({'Calories Burned': prediction_float})

if __name__ == '__main__':
    app.run(debug=True)
