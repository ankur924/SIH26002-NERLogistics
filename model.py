import pandas as pd

file_path = r"C:\Users\DELL\AppData\Local\Programs\Microsoft VS Code\sih26002_road_risk_dataset.csv"

df = pd.read_csv(file_path)

print(df.head())
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())