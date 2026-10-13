import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv("Teen_Mental_Health_Dataset (1).csv")

print(df.head())


print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


X = df.drop("depression_label", axis=1)
y = df["depression_label"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

X = pd.get_dummies(X, drop_first=True)

print("\nData after encoding:")
print(X.head())

print("\nNew shape:")
print(X.shape)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nStandardized data:")
print(X_scaled[:5])


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


print("\nPCA Result:")
print(X_pca[:5])

print("\nOriginal number of features:")
print(X.shape[1])

print("\nNumber of PCA components:")
print(X_pca.shape[1])

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Explained:")
print(sum(pca.explained_variance_ratio_))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Teen Mental Health Dataset")

plt.show()