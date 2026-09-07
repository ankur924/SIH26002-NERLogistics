import pandas as pd
import joblib

# Load trained model
model = joblib.load("road_risk_model.pkl")
preprocessor = joblib.load("road_risk_preprocessor.pkl")
label_mapping = joblib.load("risk_label_mapping.pkl")


# Different road scenarios
test_roads = pd.DataFrame([

    # 1. Normal / good road
    {
        "road_type": "Highway",
        "road_length_km": 10,
        "road_condition": 1,
        "road_width_m": 10,
        "traffic_level": 2,
        "elevation_m": 100,
        "slope_degree": 2,
        "rainfall_mm": 20,
        "temperature_c": 25,
        "previous_disruptions": 0,
        "previous_accidents": 0,
        "flood": 0,
        "landslide": 0,
        "bridge_damage": 0,
        "road_accessible": 1
    },

    # 2. Moderate traffic / moderate condition
    {
        "road_type": "Urban",
        "road_length_km": 8,
        "road_condition": 3,
        "road_width_m": 7,
        "traffic_level": 4,
        "elevation_m": 150,
        "slope_degree": 5,
        "rainfall_mm": 80,
        "temperature_c": 30,
        "previous_disruptions": 2,
        "previous_accidents": 2,
        "flood": 0,
        "landslide": 0,
        "bridge_damage": 0,
        "road_accessible": 1
    },

    # 3. High-risk road
    {
        "road_type": "Mountain",
        "road_length_km": 15,
        "road_condition": 4,
        "road_width_m": 5,
        "traffic_level": 5,
        "elevation_m": 800,
        "slope_degree": 18,
        "rainfall_mm": 180,
        "temperature_c": 20,
        "previous_disruptions": 5,
        "previous_accidents": 6,
        "flood": 1,
        "landslide": 1,
        "bridge_damage": 0,
        "road_accessible": 1
    },

    # 4. Very dangerous road
    {
        "road_type": "Mountain",
        "road_length_km": 20,
        "road_condition": 5,
        "road_width_m": 4,
        "traffic_level": 5,
        "elevation_m": 1200,
        "slope_degree": 30,
        "rainfall_mm": 300,
        "temperature_c": 18,
        "previous_disruptions": 8,
        "previous_accidents": 10,
        "flood": 1,
        "landslide": 1,
        "bridge_damage": 1,
        "road_accessible": 0
    },

    # 5. Rural road with some risk
    {
        "road_type": "Rural",
        "road_length_km": 12,
        "road_condition": 3,
        "road_width_m": 6,
        "traffic_level": 3,
        "elevation_m": 300,
        "slope_degree": 8,
        "rainfall_mm": 100,
        "temperature_c": 27,
        "previous_disruptions": 2,
        "previous_accidents": 3,
        "flood": 0,
        "landslide": 0,
        "bridge_damage": 0,
        "road_accessible": 1
    }
])


# Preprocess all test roads
processed_roads = preprocessor.transform(test_roads)

# Predictions
predictions = model.predict(processed_roads)

# Probabilities
probabilities = model.predict_proba(processed_roads)


print("\n============================================================")
print("             ROAD RISK MODEL TESTING")
print("============================================================")


for i, prediction in enumerate(predictions):

    category = label_mapping[prediction]

    print(f"\nRoad Scenario {i + 1}")
    print("-" * 40)

    print(f"Predicted Risk: {category}")

    print("\nProbabilities:")

    for class_number, probability in enumerate(probabilities[i]):
        class_name = label_mapping[class_number]
        print(f"{class_name:10}: {probability * 100:.2f}%")

print("\n============================================================")
print("                 TESTING COMPLETE")
print("============================================================")