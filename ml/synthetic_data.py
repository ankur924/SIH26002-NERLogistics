import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

# -----------------------------
# Basic road information
# -----------------------------

data = pd.DataFrame({
    "road_id": [f"R{i:05d}" for i in range(1, n + 1)],

    "road_type": np.random.choice(
        ["Highway", "State Highway", "District Road", "Village Road"],
        n,
        p=[0.15, 0.25, 0.35, 0.25]
    ),

    "road_length_km": np.round(
        np.random.uniform(0.5, 20, n), 2
    ),

    # Road condition:
    # 0 = Good, 1 = Fair, 2 = Poor, 3 = Very Poor
    "road_condition": np.random.choice(
        [0, 1, 2, 3],
        n,
        p=[0.35, 0.35, 0.20, 0.10]
    ),

    # Road width in metres
    "road_width_m": np.round(
        np.random.uniform(3, 12, n), 2
    ),

    # Traffic:
    # 0 = Low, 1 = Medium, 2 = High, 3 = Very High
    "traffic_level": np.random.choice(
        [0, 1, 2, 3],
        n,
        p=[0.30, 0.35, 0.25, 0.10]
    ),

    # Elevation
    "elevation_m": np.round(
        np.random.uniform(50, 2500, n), 1
    ),

    # Slope in degrees
    "slope_degree": np.round(
        np.random.uniform(0, 45, n), 2
    ),

    # Rainfall in mm
    "rainfall_mm": np.round(
        np.random.gamma(shape=2.5, scale=35, size=n), 2
    ),

    # Temperature
    "temperature_c": np.round(
        np.random.uniform(10, 35, n), 2
    ),

    # Historical number of disruptions
    "previous_disruptions": np.random.poisson(
        lam=2, size=n
    ),

    # Number of previous accidents
    "previous_accidents": np.random.poisson(
        lam=1.5, size=n
    )
})


# -----------------------------
# Flood information
# -----------------------------

# Flood probability increases with rainfall
flood_probability = (
    data["rainfall_mm"] / 300
)

flood_probability = np.clip(
    flood_probability, 0, 0.85
)

data["flood"] = np.random.binomial(
    1,
    flood_probability
)


# -----------------------------
# Landslide information
# -----------------------------

# Landslide probability depends on
# rainfall + slope + elevation

landslide_probability = (
    0.02
    + 0.003 * data["slope_degree"]
    + 0.0015 * data["rainfall_mm"]
    + 0.00001 * data["elevation_m"]
)

landslide_probability = np.clip(
    landslide_probability,
    0,
    0.85
)

data["landslide"] = np.random.binomial(
    1,
    landslide_probability
)


# -----------------------------
# Bridge damage
# -----------------------------

bridge_probability = 0.10

data["bridge_damage"] = np.random.binomial(
    1,
    bridge_probability,
    n
)


# -----------------------------
# Road accessibility
# -----------------------------

# 1 = Accessible
# 0 = Not accessible

accessibility_score = (
    1
    - 0.20 * data["road_condition"]
    - 0.20 * data["flood"]
    - 0.25 * data["landslide"]
    - 0.15 * data["bridge_damage"]
)

data["road_accessible"] = (
    accessibility_score > 0.35
).astype(int)


# -----------------------------
# Generate risk score
# -----------------------------

risk_score = (

    # Rainfall
    0.20 * np.clip(
        data["rainfall_mm"] / 300, 0, 1
    )

    # Flood
    + 0.20 * data["flood"]

    # Landslide
    + 0.20 * data["landslide"]

    # Road condition
    + 0.10 * (data["road_condition"] / 3)

    # Traffic
    + 0.05 * (data["traffic_level"] / 3)

    # Slope
    + 0.10 * np.clip(
        data["slope_degree"] / 45, 0, 1
    )

    # Bridge damage
    + 0.10 * data["bridge_damage"]

    # Previous disruptions
    + 0.03 * np.clip(
        data["previous_disruptions"] / 10,
        0,
        1
    )

    # Previous accidents
    + 0.02 * np.clip(
        data["previous_accidents"] / 10,
        0,
        1
    )
)


# Add small randomness so the model
# doesn't learn a perfectly deterministic rule

noise = np.random.normal(
    0,
    0.04,
    n
)

risk_score = risk_score + noise

# Keep score between 0 and 1
data["risk_score"] = np.clip(
    risk_score,
    0,
    1
)

# -----------------------------
# Risk category
# -----------------------------

data["risk_category"] = pd.cut(
    data["risk_score"],
    bins=[-0.01, 0.25, 0.50, 0.75, 1.0],
    labels=[
        "Safe",
        "Moderate",
        "High",
        "Critical"
    ]
)


# -----------------------------
# Save dataset
# -----------------------------

data.to_csv(
    "sih26002_road_risk_dataset.csv",
    index=False
)

print("Dataset created successfully!")
print("Shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nRisk distribution:")
print(data["risk_category"].value_counts())

import os 
data.to_csv("sih26002_road_risk_dataset.csv",index=False)
print("csv created successfully")
print("saved at")
print(os.path.abspath("sih26002_road_risk_dataset.csv"))