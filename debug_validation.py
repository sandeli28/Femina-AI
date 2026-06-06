import requests
import json

# The payload exactly as script.js would likely send it
# Note inputs like '1' are converted to numbers by script.js
data = {
    # Valid Integers
    "Age": 25,
    "Stress_score_0_10": 5,
    "Hormonal_Imbalance": 0,
    "Hyperandrogenism": 0,
    "Hirsutism": 0,
    "Conception_Difficulty": 0,
    "Family_History_PCOS": 0,
    "Insulin_Resistance": 0,
    "Mental_Health": 0,
    
    # Valid Floats
    "Weight_kg": 60.0,
    "Height_cm": 165.0,
    "Testosterone_ng_per_dL": 45.0,
    "LH_FSH_Ratio": 1.5,
    "Exercise_hrs_per_week": 3.0,
    "Sleep_hrs_per_night": 7.0,

    # Strings
    "Diet_Type": "Non-vegetarian",
    "Menstrual_Cycle": "Irregular",
    "Smoking_Status": "Never",

    # The Suspect: "1" converted to number 1, but model expects Optional[str]
    "Veg_or_NonVeg": 1,
    
    # Other hidden fields (int 0/1)
    "Diabetes": 0,
    "Cardiovascular_Disease": 0,
    "Childhood_Trauma": 0,
    "Family_History_Menstrual": 0
}

url = "http://127.0.0.1:8000/api/v1/predict"

try:
    print(f"Sending payload: {json.dumps(data, indent=2)}")
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
