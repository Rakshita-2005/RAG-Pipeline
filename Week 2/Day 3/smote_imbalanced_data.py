
# W2D3: Handling Imbalanced Data - SMOTE
# Feature Engineering & ML Pipeline

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from imblearn.over_sampling import SMOTE


# -----------------------------------
# 1. Create Imbalanced Dataset
# -----------------------------------

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    weights=[0.90, 0.10],
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


print("First 5 rows:")
print(df.head())


# -----------------------------------
# 2. Check Class Distribution Before SMOTE
# -----------------------------------

print("\nClass Distribution Before SMOTE:")
print(df["Target"].value_counts())


plt.figure(figsize=(6,4))

sns.countplot(
    x=df["Target"]
)

plt.title("Class Distribution Before SMOTE")

plt.xlabel("Class")
plt.ylabel("Count")


# Save image in Day 3 folder

plt.savefig(
    "before_smote.png",
    bbox_inches="tight"
)

plt.show()



# -----------------------------------
# 3. Split Dataset
# -----------------------------------

X = df.drop(
    "Target",
    axis=1
)

y = df["Target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# -----------------------------------
# 4. Apply SMOTE
# -----------------------------------

smote = SMOTE(
    random_state=42
)


X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)



print("\nClass Distribution After SMOTE:")
print(
    y_train_smote.value_counts()
)



plt.figure(figsize=(6,4))

sns.countplot(
    x=y_train_smote
)

plt.title("Class Distribution After SMOTE")

plt.xlabel("Class")
plt.ylabel("Count")


# Save image

plt.savefig(
    "after_smote.png",
    bbox_inches="tight"
)

plt.show()



# -----------------------------------
# 5. Train Model
# -----------------------------------

model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_smote,
    y_train_smote
)



# -----------------------------------
# 6. Model Prediction
# -----------------------------------

y_pred = model.predict(
    X_test
)



print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# -----------------------------------
# 7. Playground Practice Function
# -----------------------------------

def practice():

    data = [
        [10],
        [20],
        [30]
    ]

    mean_value = np.mean(data)

    return mean_value



result = practice()


print("\nPractice Output:")
print(result)


print("\nDone! Review with CIA for feedback.")