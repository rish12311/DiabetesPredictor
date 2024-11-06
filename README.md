DiabetesPredict: ML-Powered Diabetes Risk Assessment System
A machine learning-based web application that predicts diabetes risk using medical parameters. The system employs both Naive Bayes and Perceptron models to provide comprehensive risk assessments.
Features

Dual Model Prediction: Utilizes both Naive Bayes and Perceptron algorithms for comparative analysis
Real-time Assessment: Instant prediction results through REST API
User-friendly Interface: Clean and intuitive frontend for easy parameter input
Medical Parameter Analysis: Processes key health indicators including:

Glucose Level
Insulin
Body Mass Index (BMI)
Age



#Tech Stack

Python
Flask
scikit-learn
NumPy
Pandas
Flask-CORS
HTMLS/CSS


#Installation

Clone the repository

git clone https://github.com/rish12311/DiabetesPredictor.git


#Install required packages

#Run the Flask server

python app.py

#Open the frontend interface in your browser
(either open index.html on browser or use it via any framework)

#API Endpoints
Prediction Endpoint

URL: /predict
Method: POST
Request Body:

jsonCopy{
    "glucose": 140,
    "insulin": 169,
    "bmi": 32.0,
    "age": 45,
    "model": "naive_bayes"  // or "perceptron"
}

Response:

jsonCopy{
    "prediction": 1,
    "probability": {
        "non_diabetic": 0.3,
        "diabetic": 0.7
    }
}

Model Training
The models are trained on the diabetes dataset containing various medical parameters. The training process includes:

Data preprocessing and scaling
Feature selection
Model training with cross-validation
Performance metrics evaluation

