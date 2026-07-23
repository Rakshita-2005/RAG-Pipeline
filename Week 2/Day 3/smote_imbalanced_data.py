# W2D3: Handling Imbalanced Data - SMOTE

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE


# -----------------------------
# Create Imbalanced Dataset
# -----------------------------

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_classes=2,
    weights=[0.90,0.10],
    random_state=42
)


df = pd.DataFrame(
    X,
    columns=[
        "Feature1",
        "Feature2",
        "Feature3",
        "Feature4",
        "Feature5"
    ]
)

df["Target"] = y


print("Dataset:")
print(df.head())


# -----------------------------
# Check Class Distribution
# -----------------------------

print("\nClass Distribution Before SMOTE:")
print(df["Target"].value_counts())


sns.countplot(
    x=df["Target"]
)

plt.title("Before SMOTE")
plt.show()



# -----------------------------
# Train Test Split
# -----------------------------

X = df.drop("Target", axis=1)

y = df["Target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# -----------------------------
# Apply SMOTE
# -----------------------------

smote = SMOTE(
    random_state=42
)


X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\nClass Distribution After SMOTE:")
print(y_train_smote.value_counts())


sns.countplot(
    x=y_train_smote
)

plt.title("After SMOTE")
plt.show()



# -----------------------------
# Model Training
# -----------------------------

model = LogisticRegression()

model.fit(
    X_train_smote,
    y_train_smote
)


prediction = model.predict(
    X_test
)


print("\nModel Evaluation:")
print(
    classification_report(
        y_test,
        prediction
    )
)



# -----------------------------
# Playground Practice
# -----------------------------

def practice():

    values = [
        [10],
        [20],
        [30]
    ]

    result = np.mean(values)

    return result



print("\nPractice Output:")
print(practice())


print("Done! Review with CIA for feedback.")