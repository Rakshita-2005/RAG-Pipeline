# Week 3 - Day 4
# SVM & KNN Classification on Iris Dataset

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -------------------------
# Load Dataset
# -------------------------
iris = load_iris()

X = iris.data
y = iris.target

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# -------------------------
# Feature Scaling
# (Important for SVM & KNN)
# -------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Support Vector Machine
# -------------------------
svm = SVC(kernel="rbf", random_state=42)

svm.fit(X_train_scaled, y_train)

svm_pred = svm.predict(X_test_scaled)

print("=" * 50)
print("Support Vector Machine")
print("=" * 50)
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# -------------------------
# K-Nearest Neighbors
# -------------------------
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

knn_pred = knn.predict(X_test_scaled)

print("=" * 50)
print("K-Nearest Neighbors")
print("=" * 50)
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# -------------------------
# Confusion Matrix
# -------------------------
fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    svm_pred,
    display_labels=iris.target_names,
    ax=ax,
)

plt.title("SVM Confusion Matrix")
plt.savefig("output.png")
plt.show()

# -------------------------
# Accuracy Comparison
# -------------------------
results = pd.DataFrame({
    "Model": ["SVM", "KNN"],
    "Accuracy": [
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, knn_pred),
    ]
})

print("\nModel Comparison")
print(results)