# Week 3 - Day 5
# Hyperparameter Tuning using GridSearchCV & RandomizedSearchCV

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# --------------------------
# Load Dataset
# --------------------------
iris = load_iris()

X = iris.data
y = iris.target

# --------------------------
# Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# --------------------------
# Feature Scaling
# --------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# Base Model
# --------------------------
svm = SVC()

# --------------------------
# Grid Search
# --------------------------
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
}

grid = GridSearchCV(
    svm,
    param_grid,
    cv=5,
    scoring="accuracy",
)

grid.fit(X_train, y_train)

print("=" * 50)
print("GRID SEARCH")
print("=" * 50)
print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

grid_pred = grid.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, grid_pred))

# --------------------------
# Random Search
# --------------------------
param_random = {
    "C": randint(1, 20),
    "kernel": ["linear", "rbf"],
}

random = RandomizedSearchCV(
    svm,
    param_random,
    n_iter=10,
    cv=5,
    random_state=42,
)

random.fit(X_train, y_train)

print("\n" + "=" * 50)
print("RANDOM SEARCH")
print("=" * 50)
print("Best Parameters:", random.best_params_)
print("Best CV Score:", random.best_score_)

random_pred = random.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, random_pred))

# --------------------------
# Comparison Plot
# --------------------------
scores = [
    accuracy_score(y_test, grid_pred),
    accuracy_score(y_test, random_pred),
]

labels = ["Grid Search", "Random Search"]

plt.figure(figsize=(6, 5))
plt.bar(labels, scores)
plt.ylabel("Accuracy")
plt.title("GridSearchCV vs RandomizedSearchCV")
plt.ylim(0.8, 1.0)

plt.savefig("output.png")
plt.show()