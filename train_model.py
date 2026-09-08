import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = r"C:\Users\DELL\AppData\Local\Programs\Microsoft VS Code\sih26002_road_risk_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)



df = df.drop(columns=["road_id", "risk_score"])



X = df.drop(columns=["risk_category"])

y = df["risk_category"]




categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns


print("\nCategorical Columns:")
print(list(categorical_columns))

print("\nNumerical Columns:")
print(list(numerical_columns))




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




model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)



pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))




print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed!")




y_pred = pipeline.predict(X_test)




accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nRisk Category Distribution:")
print(df["risk_category"].value_counts())

print("\nRisk Category Percentage:")
print(df["risk_category"].value_counts(normalize=True) * 100)