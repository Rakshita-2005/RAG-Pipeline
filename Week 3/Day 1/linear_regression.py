# W3D1: Linear Regression - Scikit Learn
# Train Linear, Ridge and Lasso Regression models
# Evaluate and compare performance


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# Load real dataset
data = fetch_california_housing()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target)


print("Dataset Shape:")
print(X.shape)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



# Models
models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(alpha=1.0),

    "Lasso Regression":
        Lasso(alpha=0.001)

}



results = []


for name, model in models.items():

    # Train
    model.fit(
        X_train_scaled,
        y_train
    )


    # Prediction
    predictions = model.predict(
        X_test_scaled
    )


    # Evaluation metrics
    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    results.append(
        [
            name,
            mse,
            rmse,
            mae,
            r2
        ]
    )


    print("\n", name)

    print("Coefficients:")
    print(model.coef_)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)



# Results table

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MSE",
        "RMSE",
        "MAE",
        "R2"
    ]
)


print("\nModel Comparison:")
print(results_df)


results_df.to_csv(
    "outputs/results_table.csv",
    index=False
)



# Predicted vs Actual plot

best_model = LinearRegression()

best_model.fit(
    X_train_scaled,
    y_train
)

predictions = best_model.predict(
    X_test_scaled
)


plt.figure(figsize=(8,5))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel(
    "Actual Values"
)

plt.ylabel(
    "Predicted Values"
)

plt.title(
    "Predicted vs Actual"
)

plt.savefig(
    "outputs/predicted_vs_actual.png"
)

plt.close()



# Residual plot

residuals = y_test - predictions


plt.figure(figsize=(8,5))

plt.scatter(
    predictions,
    residuals
)

plt.axhline(
    y=0
)

plt.xlabel(
    "Predicted Values"
)

plt.ylabel(
    "Residuals"
)

plt.title(
    "Residual Plot"
)

plt.savefig(
    "outputs/residual_plot.png"
)

plt.close()



print("\nCompleted Successfully!")