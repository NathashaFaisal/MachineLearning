# T6 - Logistic Regression

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("Teen_Mental_Health_Dataset (1).csv")

# 2. Features and target
X = df.drop("depression_label", axis=1)
y = df["depression_label"]

# 3. Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. Standardize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Create model
model = LogisticRegression(max_iter=1000)

# 7. Train
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Results
print("LOGISTIC REGRESSION")
print("-------------------")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))