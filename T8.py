
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv("Teen_Mental_Health_Dataset (1).csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


X = df.drop("depression_label", axis=1)
y = df["depression_label"]

# Convert categorical columns to numerical
X = pd.get_dummies(X, drop_first=True)

print("\nShape after encoding:")
print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


model = GaussianNB()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nPredicted values:")
print(y_pred)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))