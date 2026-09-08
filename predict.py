import pandas as pd
import joblib



model = joblib.load(
    "road_risk_smote_xgboost.pkl"
)

preprocessor = joblib.load(
    "road_risk_preprocessor.pkl"
)



road = {
    "road_type": "State Highway",
    "road_length_km": 10.5,
    "road_condition": 3,
    "road_width_m": 7.0,
    "traffic_level": 4,
    "elevation_m": 250,
    "slope_degree": 8,
    "rainfall_mm": 180,
    "temperature_c": 28,
    "previous_disruptions": 2,
    "previous_accidents": 3,
    "flood": 1,
    "landslide": 0,
    "bridge_damage": 0,
    "road_accessible": 1
}




input_data = pd.DataFrame([road])




input_processed = preprocessor.transform(
    input_data
)



prediction = model.predict(
    input_processed
)



labels = {
    0: "Safe",
    1: "Moderate",
    2: "High",
    3: "Critical"
}

risk_category = labels[int(prediction[0])]


print("\n==============================")
print("ROAD RISK PREDICTION")
print("==============================")

print("Risk Category:", risk_category)




probabilities = model.predict_proba(
    input_processed
)[0]

print("\nRisk Probabilities:")

for i, probability in enumerate(probabilities):

    print(
        f"{labels[i]}: {probability * 100:.2f}%"
    )