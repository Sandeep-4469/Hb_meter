import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# 1. Load data
data_path = '/home/roshan-raj/.gemini/antigravity/scratch/cleaned_clinical_data.csv'
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path)

# Drop rows where ground truth hb_actual or device output hb_meter is missing
df = df.dropna(subset=['hb_actual', 'hb_meter'])

# Encode Gender as binary (0=Female, 1=Male)
df['Gender_bin'] = df['Gender'].map({'F': 0, 'M': 1})
df['Gender_bin'] = df['Gender_bin'].fillna(0)

# Fill missing Age, RBC, MCV, MCH with median
df['Age'] = df['Age'].fillna(df['Age'].median())
for col in ['RBC', 'MCV', 'MCH']:
    df[col] = df[col].fillna(df[col].median())

print(f"Loaded {len(df)} samples for training.")

# Define target
target = 'hb_actual'

# 2. Train Clinical Calibration Model (Hb_Meter + Age + Gender + RBC + MCV + MCH)
features_clinical = ['hb_meter', 'Age', 'Gender_bin', 'RBC', 'MCV', 'MCH']
X_cl = df[features_clinical].values
y_cl = df[target].values

model_clinical = LinearRegression()
model_clinical.fit(X_cl, y_cl)

# 3. Train Non-Invasive Calibration Model (Hb_Meter + Age + Gender)
features_ni = ['hb_meter', 'Age', 'Gender_bin']
X_ni = df[features_ni].values
y_ni = df[target].values

model_ni = LinearRegression()
model_ni.fit(X_ni, y_ni)

# 4. Save models to the Hb_meter folder
output_dir = '/home/roshan-raj/Hb_meter'
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model_clinical, os.path.join(output_dir, 'calibration_clinical.joblib'))
joblib.dump(model_ni, os.path.join(output_dir, 'calibration_non_invasive.joblib'))

print("✅ Calibration models successfully trained and saved:")
print(f"  - Clinical: {os.path.join(output_dir, 'calibration_clinical.joblib')}")
print(f"  - Non-Invasive: {os.path.join(output_dir, 'calibration_non_invasive.joblib')}")
print("\nModel Coefficients:")
print(f"  Clinical features: {features_clinical}")
print(f"  Clinical coefs: {model_clinical.coef_}")
print(f"  Clinical intercept: {model_clinical.intercept_}")
print(f"  Non-Invasive features: {features_ni}")
print(f"  Non-Invasive coefs: {model_ni.coef_}")
print(f"  Non-Invasive intercept: {model_ni.intercept_}")
