from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load models
with open("naive_bayes_model.pkl", "rb") as f:
    naive_bayes_model = pickle.load(f)

with open("perceptron_model.pkl", "rb") as f:
    perceptron_model = pickle.load(f)

# Load the scaler that was used during training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def validate_input(data):
    """Validate input ranges based on typical medical values"""
    if not (0 <= data['glucose'] <= 500):  # Normal range plus margin
        raise ValueError("Glucose should be between 0 and 500 mg/dL")
    if not (0 <= data['insulin'] <= 1000):  # Normal range plus margin
        raise ValueError("Insulin should be between 0 and 1000 μU/mL")
    if not (10 <= data['bmi'] <= 100):  # Normal range plus margin
        raise ValueError("BMI should be between 10 and 100")
    if not (0 <= data['age'] <= 120):
        raise ValueError("Age should be between 0 and 120 years")

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Check if required fields are present
        required_fields = ['glucose', 'insulin', 'bmi', 'age', 'model']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Validate input ranges
        try:
            validate_input(data)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Prepare features
        features = np.array([[
            data['glucose'],
            data['insulin'],
            data['bmi'],
            data['age']
        ]])

        # Scale the features using the same scaler used during training
        features_scaled = scaler.transform(features)

        # Choose the model
        if data['model'] == 'naive_bayes':
            model = naive_bayes_model
        elif data['model'] == 'perceptron':
            model = perceptron_model
        else:
            return jsonify({"error": "Invalid model type. Use 'naive_bayes' or 'perceptron'"}), 400

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        return jsonify({
            "prediction": int(prediction),
            "probability": {
                "non_diabetic": float(prediction_proba[0]),
                "diabetic": float(prediction_proba[1])
            }
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)