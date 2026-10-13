# T5 - Multiple Linear Regression

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("Teen_Mental_Health_Dataset (1).csv")

# 2. Select multiple input variables
features = [
    "age",
    "daily_social_media_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level"
]

X = df[features]
y = df["sleep_hours"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Create model
model = LinearRegression()

# 5. Train
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MULTIPLE LINEAR REGRESSION")
print("--------------------------")

print("\nCoefficients:")
for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)
print("MSE:", mse)
print("R2 Score:", r2)