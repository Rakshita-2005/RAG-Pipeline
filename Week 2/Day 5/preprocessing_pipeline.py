# W2D5: End-to-End Preprocessing Pipeline

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def create_pipeline():

    # Sample dataset
    data = {
        "Age": [25, 30, None, 40, 35],
        "Salary": [30000, 50000, 45000, None, 70000],
        "City": ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"],
        "Purchased": [0, 1, 0, 1, 1]
    }

    df = pd.DataFrame(data)

    print("Original Data:")
    print(df)


    # Features and target
    X = df.drop("Purchased", axis=1)
    y = df["Purchased"]


    # Numerical columns
    numerical_features = [
        "Age",
        "Salary"
    ]

    # Categorical columns
    categorical_features = [
        "City"
    ]


    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer",
             SimpleImputer(strategy="mean")),

            ("scaler",
             StandardScaler())
        ]
    )


    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer",
             SimpleImputer(strategy="most_frequent")),

            ("encoder",
             OneHotEncoder(handle_unknown="ignore"))
        ]
    )


    # Combine preprocessing
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # Apply preprocessing
    X_train_processed = preprocessing.fit_transform(X_train)

    X_test_processed = preprocessing.transform(X_test)


    print("\nProcessed Training Data:")
    print(X_train_processed)


    print("\nProcessed Testing Data:")
    print(X_test_processed)


    return X_train_processed, X_test_processed



if __name__ == "__main__":

    create_pipeline()

    print("\nPipeline execution completed successfully!")