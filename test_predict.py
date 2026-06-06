
import requests
import json

url = 'http://127.0.0.1:5000/predict'

# Optimized Data Payload (Omitting optional fields like Diet, Diabetes history etc.)
data = {
    # Required
    "Age": 25,
    "Weight_kg": 65.5,
    "Height_cm": 160.0,
    "Testosterone_ng_per_dL": 45.0,
    "LH_FSH_Ratio": 1.5,
    "Exercise_hrs_per_week": 5.0,
    "Sleep_hrs_per_night": 7.5,
    "Stress_score_0_10": 5,
    "Menstrual_Cycle": "Irregular",
    "Hormonal_Imbalance": 1,
    "Hyperandrogenism": 0,
    "Hirsutism": 0,
    "Conception_Difficulty": 0,
    "Family_History_PCOS": 1,
    "Insulin_Resistance": 0,
    "Mental_Health": 0,
    
    # Optional Fields from RFE (We intentionally OMIT some to test optionality)
    # "Diabetes": 0, 
    # "Cardiovascular_Disease": 0,
    # "Childhood_Trauma": 0,
    # "Family_History_Menstrual": 0,
    # "Diet_Type": "Mixed",
    # "Veg_or_NonVeg": "Non-vegetarian",
    
    # Kept this one
    "Smoking_Status": "Never"
}

print("Sending request with REDUCED feature set (testing optimization)...")
try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("\nResponse JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Request failed: {e}")
