# W2D2 Feature Scaling & Selection

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.feature_selection import SelectKBest, f_classif


# -----------------------------
# Create Dataset
# -----------------------------

data = {
    "Age": [22,25,31,45,52,36,29,41,55,23],
    "Salary": [25000,32000,45000,65000,90000,52000,38000,72000,110000,28000],
    "Department": [
        "IT","HR","Finance","IT",
        "Marketing","Finance","IT",
        "HR","Marketing","IT"
    ],
    "Experience": [
        "Low","Medium","Medium",
        "High","High","Medium",
        "Low","High","High","Low"
    ],
    "Purchased":[0,1,0,1,1,1,0,1,1,0]
}


df = pd.DataFrame(data)

print(df)


# -----------------------------
# Encoding
# -----------------------------

# Label Encoder

le = LabelEncoder()

df["Department_Label"] = le.fit_transform(
    df["Department"]
)


# One Hot Encoder

ohe = OneHotEncoder(
    sparse_output=False
)

department_encoded = ohe.fit_transform(
    df[["Department"]]
)

department_df = pd.DataFrame(
    department_encoded,
    columns=ohe.get_feature_names_out(["Department"])
)

print("\nOne Hot Encoding:")
print(department_df)


# Ordinal Encoder

oe = OrdinalEncoder(
    categories=[["Low","Medium","High"]]
)

df["Experience_encoded"] = oe.fit_transform(
    df[["Experience"]]
)


print("\nEncoded Dataset:")
print(df)



# -----------------------------
# Feature Scaling
# -----------------------------

numeric_features = df[
    ["Age","Salary"]
]


# Before Scaling

numeric_features.hist(
    figsize=(8,4)
)

plt.title("Before Scaling")
plt.show()



# StandardScaler

standard = StandardScaler()

standard_data = standard.fit_transform(
    numeric_features
)

pd.DataFrame(
    standard_data,
    columns=["Age","Salary"]
).hist(figsize=(8,4))

plt.title("StandardScaler")
plt.show()



# MinMaxScaler

minmax = MinMaxScaler()

minmax_data = minmax.fit_transform(
    numeric_features
)


pd.DataFrame(
    minmax_data,
    columns=["Age","Salary"]
).hist(figsize=(8,4))

plt.title("MinMaxScaler")
plt.show()



# RobustScaler

robust = RobustScaler()

robust_data = robust.fit_transform(
    numeric_features
)


pd.DataFrame(
    robust_data,
    columns=["Age","Salary"]
).hist(figsize=(8,4))

plt.title("RobustScaler")
plt.show()



# -----------------------------
# SelectKBest Feature Selection
# -----------------------------


X = df.drop(
    ["Purchased","Experience","Department"],
    axis=1
)

y = df["Purchased"]


selector = SelectKBest(
    score_func=f_classif,
    k=5
)


X_selected = selector.fit_transform(
    X,y
)


selected_features = X.columns[
    selector.get_support()
]


print("\nTop 5 Features:")
print(selected_features)



# -----------------------------
# Playground Practice Function
# -----------------------------

def practice():

    values = np.array(
        [[10],[20],[30]]
    )

    scaler = StandardScaler()

    result = scaler.fit_transform(values)

    return result



output = practice()

print("\nPractice Output:")
print(output)

print("Done! Review with CIA for feedback.")