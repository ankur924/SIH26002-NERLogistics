import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# =========================
# 1. LOAD DATA
# =========================

file_path = r"C:\Users\DELL\AppData\Local\Programs\Microsoft VS Code\sih26002_road_risk_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)


# =========================
# 2. REMOVE ID + RISK SCORE
# =========================

df = df.drop(columns=["road_id", "risk_score"])


# =========================
# 3. FEATURES + TARGET
# =========================

X = df.drop(columns=["risk_category"])

y = df["risk_category"]


# =========================
# 4. ENCODE TARGET
# =========================

target_mapping = {
    "Safe": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3
}

y = y.map(target_mapping)


# =========================
# 5. COLUMN TYPES
# =========================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns


# =========================
# 6. PREPROCESSING
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# =========================
# 7. XGBOOST
# =========================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42
)


# =========================
# 8. PIPELINE
# =========================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# =========================
# 9. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 10. TRAIN
# =========================

print("\nTraining XGBoost...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# =========================
# 11. PREDICTION
# =========================

y_pred = pipeline.predict(X_test)


# =========================
# 12. EVALUATION
# =========================

print("\n==============================")
print("XGBOOST RESULTS")
print("==============================")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Safe",
            "Moderate",
            "High",
            "Critical"
        ]
    )
)


print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))