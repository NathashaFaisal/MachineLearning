# T3 - Cross Validation

import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. Load dataset
df = pd.read_csv("Teen_Mental_Health_Dataset (1).csv")

# 2. Features and target
X = df.drop("depression_label", axis=1)
y = df["depression_label"]

# 3. Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# 4. Create model pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. K-Fold
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 6. Cross validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("K-FOLD CROSS VALIDATION")
print("-----------------------")

print("Fold Scores:")
print(scores)

print("\nMean Accuracy:")
print(scores.mean())

print("\nStandard Deviation:")
print(scores.std())