import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier




file_path = r"C:\Users\DELL\AppData\Local\Programs\Microsoft VS Code\sih26002_road_risk_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)




df = df.drop(columns=["road_id", "risk_score"])


X = df.drop(columns=["risk_category"])
y = df["risk_category"]




target_mapping = {
    "Safe": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3
}

y = y.map(target_mapping)



categorical_columns = X.select_dtypes(
    include=["object"]
).columns



preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


print("\nBefore SMOTE:")
print(pd.Series(y_train).value_counts().sort_index())



smote = SMOTE(
    sampling_strategy="not majority",
    k_neighbors=3,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts().sort_index())


# =========================
# 10. XGBOOST
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
# 11. TRAIN
# =========================

print("\nTraining SMOTE + XGBoost...")

model.fit(
    X_train_smote,
    y_train_smote
)

print("Training completed!")



y_pred = model.predict(X_test)




print("\n==============================")
print("SMOTE + XGBOOST RESULTS")
print("==============================")


print("\nAccuracy:")

print(
    accuracy_score(
        y_test,
        y_pred
    )
)


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

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)




joblib.dump(
    model,
    "road_risk_smote_xgboost.pkl"
)

joblib.dump(
    preprocessor,
    "road_risk_preprocessor.pkl"
)

print("\nModel saved successfully!")