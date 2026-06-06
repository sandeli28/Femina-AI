
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import scipy.stats
from flask_cors import CORS
from pydantic import BaseModel, Field, ValidationError
from flasgger import Swagger, swag_from
import os
from typing import Optional
from doctor_service import get_nearby_doctors

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Load the saved model
MODEL_PATH = 'pcos_nonlinear_stack_calibrated.joblib'
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define expected feature columns
FEATURE_COLUMNS = [
    'Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism', 'Testosterone_ng_per_dL',
    'Menstrual_Cycle', 'LH_FSH_Ratio', 'Conception_Difficulty', 'Family_History_PCOS',
    'Family_History_Menstrual', 'Insulin_Resistance', 'Diabetes', 'Cardiovascular_Disease',
    'Mental_Health', 'Childhood_Trauma', 'Age', 'Weight_kg', 'Height_cm',
    'Diet_Type', 'Veg_or_NonVeg', 'Exercise_hrs_per_week', 'Sleep_hrs_per_night',
    'Stress_score_0_10', 'Smoking_Status'
]

# Pydantic Model for Input Validation
# Optimized based on RFE Analysis: Low importance fields are now Optional
class PCOSInput(BaseModel):
    # Numerical
    Age: int = Field(..., ge=10, le=100, description="Age in years (10-100)")
    Weight_kg: float = Field(..., gt=20, lt=200, description="Weight in kg")
    Height_cm: float = Field(..., gt=100, lt=250, description="Height in cm")
    Testosterone_ng_per_dL: float = Field(..., ge=0, description="Testosterone level")
    LH_FSH_Ratio: float = Field(..., ge=0, description="LH/FSH Ratio")
    Exercise_hrs_per_week: float = Field(..., ge=0, le=168, description="Hours of exercise per week")
    Sleep_hrs_per_night: float = Field(..., ge=0, le=24, description="Hours of sleep per night")
    Stress_score_0_10: int = Field(..., ge=0, le=10, description="Self-reported stress (0-10)")

    # Categorical / Binary (0 or 1)
    Hormonal_Imbalance: int = Field(..., ge=0, le=1)
    Hyperandrogenism: int = Field(..., ge=0, le=1)
    Hirsutism: int = Field(..., ge=0, le=1)
    Conception_Difficulty: int = Field(..., ge=0, le=1)
    Family_History_PCOS: int = Field(..., ge=0, le=1)
    Insulin_Resistance: int = Field(..., ge=0, le=1)
    Mental_Health: int = Field(..., ge=0, le=1)
    
    # Optional Fields (Low Importance per RFE)
    Diabetes: Optional[int] = Field(None, ge=0, le=1, description="Optional: History of Diabetes")
    Cardiovascular_Disease: Optional[int] = Field(None, ge=0, le=1, description="Optional: CVD History")
    Childhood_Trauma: Optional[int] = Field(None, ge=0, le=1, description="Optional")
    Family_History_Menstrual: Optional[int] = Field(None, ge=0, le=1, description="Optional")
    
    # Smoking status ranked 17th but kept as required for now, or could be optional
    Smoking_Status: Optional[str] = Field(None, description="Never, Former, Current") # Default to None if missing

    # Strings
    Menstrual_Cycle: str = Field(..., description="Regular, Irregular, Oligomenorrhea, Amenorrhea")
    
    # Optional Strings
    Diet_Type: Optional[str] = Field(None, description="Optional: Vegetarian, Non-vegetarian, Mixed, Vegan")
    Veg_or_NonVeg: Optional[str] = Field(None, description="Optional")


def get_recommendations(data):
    recs = []
    if data.get('Age', 0) > 35:
        recs.append("Since you are over 35, regular checkups specifically for metabolic health are recommended.")
    if data.get('Weight_kg', 0) > 80: # Simple threshold logic
        recs.append("Maintaining a healthy weight through diet and exercise can significantly improve PCOS symptoms.")
    if data.get('Insulin_Resistance', 0) == 1:
        recs.append("Monitor blood sugar levels and consider a low-glycemic index diet to manage insulin resistance.")
    if data.get('Stress_score_0_10', 0) > 7:
        recs.append("High stress can exacerbate PCOS symptoms. Consider stress management techniques like yoga or meditation.")
    if not recs:
        recs.append("Maintain a healthy lifestyle with balanced diet and regular exercise.")
    return recs

def calculate_uncertainty(probabilities):
    """
    Calculate uncertainty using Entropy.
    Returns: entropy score (0 to ~0.69)
    """
    try:
        # Clip probabilities to avoid log(0)
        probs = np.clip(probabilities, 1e-10, 1 - 1e-10)
        entropy = scipy.stats.entropy(probs, axis=1)[0]
        return float(entropy)
    except Exception as e:
        print(f"Uncertainty calculation error: {e}")
        return 0.0

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")

@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
@swag_from({
    'responses': {
        200: {
            'description': 'Prediction Result',
            'schema': {
                'type': 'object',
                'properties': {
                    'prediction': {'type': 'integer'},
                    'probability': {'type': 'number'},
                    'risk': {'type': 'string'},
                    'risk_class': {'type': 'string'},
                    'confidence_score': {'type': 'string', 'example': 'High Confidence'},
                    'uncertainty_metric': {'type': 'number', 'example': 0.12},
                    'recommendations': {'type': 'array', 'items': {'type': 'string'}}
                }
            }
        }
    },
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'id': 'PCOSInput',
                'required': ['Age', 'Weight_kg', 'Height_cm', 'Menstrual_Cycle'],
                'properties': {
                    'Age': {'type': 'integer', 'example': 25},
                    'Weight_kg': {'type': 'number', 'example': 65.5},
                    'Height_cm': {'type': 'number', 'example': 160.0},
                    'Testosterone_ng_per_dL': {'type': 'number', 'example': 45.0},
                    'Menstrual_Cycle': {'type': 'string', 'example': 'Irregular'},
                    'Hormonal_Imbalance': {'type': 'integer', 'example': 1},
                     # ... others ...
                }
            }
        }
    ]
})
def predict():
    """
    PCOS Prediction Endpoint
    Predicts the likelihood of PCOS based on health and lifestyle factors.
    """
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Validate input using Pydantic
        input_data = PCOSInput(**request.json)
        
        # Convert Pydantic model to dict, then to DataFrame
        data_dict = input_data.dict()
        
        # Ensure column order matches training. Missing optional fields will be None/NaN
        ordered_data = {col: data_dict.get(col) for col in FEATURE_COLUMNS}
        input_df = pd.DataFrame([ordered_data])
        
        # Predict
        # Note: The model pipeline includes SimpleImputer which will handle None/NaN values from optional fields
        probas = model.predict_proba(input_df)
        prob = probas[:, 1][0]
        prediction = int(prob >= 0.5)
        
        # Calculate Uncertainty
        entropy = calculate_uncertainty(probas)
        max_entropy = np.log(2) # approx 0.693
        
        # Define Confidence Status
        if entropy < 0.3:
            conf_status = "High Confidence"
        elif entropy < 0.5:
            conf_status = "Moderate Confidence"
        else:
            conf_status = "Uncertain/Ambiguous"

        # Risk categorization
        if prob >= 0.75:
            risk = "High Risk"
            risk_class = "high"
        elif prob >= 0.35:
            risk = "Moderate Risk"
            risk_class = "moderate"
        else:
            risk = "Low Risk"
            risk_class = "low"
            
        # Get Recommendations
        recommendations = get_recommendations(data_dict)
        
        return jsonify({
            "prediction": prediction,
            "probability": round(prob * 100, 2),
            "risk": risk,
            "risk_class": risk_class,
            "confidence_score": conf_status,
            "uncertainty_metric": round(entropy, 4),
            "recommendations": recommendations
        })

    except ValidationError as e:
        print(f"Validation Error: {e.errors()}")
        return jsonify({"error": "Validation Error", "details": e.errors()}), 400
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/get_nearby_doctors", methods=["GET"])
def doctors_api():
    """
    API to fetch nearby doctors using OSM/Scraping service
    """
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        doctors = get_nearby_doctors(lat, lon)
        return jsonify(doctors)
    except Exception as e:
        print(f"Doctor API Error: {e}")
        return jsonify([])

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
