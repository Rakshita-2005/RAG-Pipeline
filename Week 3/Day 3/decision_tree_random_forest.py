# Week 3 - Day 3
# Decision Trees & Random Forests
# Author: Rakshita Kurahatti

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
iris = load_iris()

X = iris.data
y = iris.target

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Decision Tree Model
# -----------------------------
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

print("="*50)
print("Decision Tree Accuracy")
print("="*50)
print(accuracy_score(y_test, dt_predictions))

print("\nClassification Report")
print(classification_report(y_test, dt_predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_predictions))

# -----------------------------
# Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n" + "="*50)
print("Random Forest Accuracy")
print("="*50)
print(accuracy_score(y_test, rf_predictions))

print("\nClassification Report")
print(classification_report(y_test, rf_predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_predictions))

# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.Series(
    rf_model.feature_importances_,
    index=iris.feature_names
)

print("\nFeature Importance")
print(importance.sort_values(ascending=False))

# -----------------------------
# Plot Decision Tree
# -----------------------------
plt.figure(figsize=(12,8))

plot_tree(
    dt_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree")

plt.savefig("output.png")

plt.show()