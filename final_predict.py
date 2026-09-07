import pandas as pd
import joblib

# Load trained model and preprocessing files
model = joblib.load("road_risk_model.pkl")
preprocessor = joblib.load("road_risk_preprocessor.pkl")
label_mapping = joblib.load("risk_label_mapping.pkl")

# New road data
road = pd.DataFrame([{
    "road_type": "Highway",
    "road_length_km": 12.5,
    "road_condition": 3,
    "road_width_m": 8.5,
    "traffic_level": 4,
    "elevation_m": 250,
    "slope_degree": 5,
    "rainfall_mm": 120,
    "temperature_c": 28,
    "previous_disruptions": 2,
    "previous_accidents": 3,
    "flood": 1,
    "landslide": 0,
    "bridge_damage": 0,
    "road_accessible": 1
}])

# Preprocess input
road_processed = preprocessor.transform(road)

# Predict
prediction = model.predict(road_processed)[0]

# Convert number back to category
risk_category = label_mapping[prediction]

print("\n==============================")
print("     ROAD RISK PREDICTION")
print("==============================")

print(f"\nRisk Category: {risk_category}")

# Probability for each class
probabilities = model.predict_proba(road_processed)[0]

print("\nRisk Probabilities:")

for class_number, probability in enumerate(probabilities):
    category = label_mapping[class_number]
    print(f"{category}: {probability * 100:.2f}%")

print("==============================")