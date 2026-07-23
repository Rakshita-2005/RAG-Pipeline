# Week 2 Day 4
# Train/Test Split and Cross Validation

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


# Load dataset
data = load_iris()

X = data.data
y = data.target


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)



# -----------------------------
# Train Model
# -----------------------------

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)

print("Test Accuracy:", accuracy)



# -----------------------------
# Cross Validation
# -----------------------------

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)


print("\nCross Validation Scores:")
print(scores)


print("\nAverage CV Accuracy:")
print(scores.mean())



# Practice Function

def practice():

    numbers = [10,20,30,40,50]

    train, test = train_test_split(
        numbers,
        test_size=0.2,
        random_state=1
    )

    print("Train:", train)
    print("Test:", test)



practice()

print("Done! Review with CIA for feedback.")