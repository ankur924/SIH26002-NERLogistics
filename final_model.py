import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier



file_path = r"C:\Users\DELL\AppData\Local\Programs\Microsoft VS Code\sih26002_road_risk_dataset.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("SIH26002 ROAD RISK PREDICTION MODEL")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)




print("\nOriginal Risk Category Distribution:")
print(df["risk_category"].value_counts())

print("\nRisk Category Percentage:")
print(
    df["risk_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)




df = df.drop(
    columns=["road_id", "risk_score"]
)



X = df.drop(
    columns=["risk_category"]
)

y = df["risk_category"]




target_mapping = {
    "Safe": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3
}

reverse_mapping = {
    0: "Safe",
    1: "Moderate",
    2: "High",
    3: "Critical"
}

y = y.map(target_mapping)


categorical_columns = X.select_dtypes(
    include=["object"]
).columns

print("\nCategorical Features:")
print(list(categorical_columns))



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

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))



X_train = preprocessor.fit_transform(
    X_train
)

X_test = preprocessor.transform(
    X_test
)




print("\nClass Distribution Before SMOTE:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)


smote = SMOTE(
    sampling_strategy="not majority",
    k_neighbors=3,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\nClass Distribution After SMOTE:")
print(
    pd.Series(y_train_smote)
    .value_counts()
    .sort_index()
)


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




print("\nTraining final model...")

model.fit(
    X_train_smote,
    y_train_smote
)

print("Training completed!")




y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
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
        ],
        digits=4
    )
)


print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


joblib.dump(
    model,
    "road_risk_model.pkl"
)

joblib.dump(
    preprocessor,
    "road_risk_preprocessor.pkl"
)

joblib.dump(
    reverse_mapping,
    "risk_label_mapping.pkl"
)




feature_information = {
    "features": list(X.columns),
    "categorical_features": list(categorical_columns),
    "target": "risk_category",
    "target_classes": [
        "Safe",
        "Moderate",
        "High",
        "Critical"
    ]
}

joblib.dump(
    feature_information,
    "road_risk_features.pkl"
)



print("\n" + "=" * 60)
print("MODEL FILES SAVED")
print("=" * 60)

print("\n1. road_risk_model.pkl")
print("2. road_risk_preprocessor.pkl")
print("3. risk_label_mapping.pkl")
print("4. road_risk_features.pkl")

print("\nSIH26002 ML MODEL TRAINING COMPLETE!")