# T1 - Data Preprocessing

import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("Teen_Mental_Health_Dataset (1).csv")

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATA TYPES")
print(df.dtypes)

# 2. Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 3. Separate features and target
X = df.drop("depression_label", axis=1)
y = df["depression_label"]

# 4. Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

print("\nSHAPE AFTER ENCODING")
print(X.shape)

# 5. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nSTANDARDIZED DATA")
print(X_scaled[:5])

print("\nPREPROCESSING COMPLETED")