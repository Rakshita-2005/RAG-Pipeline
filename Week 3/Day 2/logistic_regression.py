# W3D2: Logistic Regression & Classification
# Train Logistic Regression on a real dataset
# Evaluate using Accuracy, Precision, Recall, F1 Score
# Save confusion matrix and classification report in the same folder


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------
# 1. Load Real Dataset
# ------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target)


print("Dataset Shape:")
print(X.shape)


# ------------------------------------------------
# 2. Split Dataset
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# 3. Feature Scaling
# ------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ------------------------------------------------
# 4. Train Logistic Regression Model
# ------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------------
# 5. Prediction
# ------------------------------------------------

y_pred = model.predict(
    X_test_scaled
)


# ------------------------------------------------
# 6. Model Evaluation
# ------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\nModel Evaluation")

print("----------------------")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)



# ------------------------------------------------
# 7. Confusion Matrix
# ------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix:")

print(cm)



# ------------------------------------------------
# 8. Save Classification Report
# ------------------------------------------------

report = classification_report(
    y_test,
    y_pred
)


with open(
    "classification_report.txt",
    "w"
) as file:

    file.write(report)



print("\nClassification report saved successfully!")



# ------------------------------------------------
# 9. Plot Confusion Matrix
# ------------------------------------------------

plt.figure(
    figsize=(6,5)
)


plt.imshow(cm)


plt.title(
    "Confusion Matrix - Logistic Regression"
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "Actual Label"
)


plt.colorbar()


# Add values inside matrix

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.savefig(
    "confusion_matrix.png"
)


plt.close()



print("Confusion matrix saved successfully!")


print("\nCompleted Successfully!")