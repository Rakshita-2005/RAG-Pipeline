# W1D4: Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def practice():
    # Sample dataset
    data = {
        "Age": [21, 25, 30, 35, 40, 45],
        "Salary": [25000, 35000, 50000, 60000, 75000, 90000],
        "Department": ["IT", "HR", "IT", "Finance", "IT", "HR"]
    }

    # Load dataset
    df = pd.DataFrame(data)

    print("Dataset Information:")
    df.info()

    print("\nStatistical Description:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Observations
    print("\nEDA Observations:")
    print("""
    1. Dataset contains employee age, salary and department details.
    2. Salary increases with employee age in this sample.
    3. IT department has the highest number of records.
    4. No missing values are present.
    5. Salary distribution contains a wide range of values.
    """)

    # Numeric column distributions
    df.hist(figsize=(8,4))
    plt.show()

    # Correlation heatmap
    correlation = df.select_dtypes(
        include="number"
    ).corr()

    sns.heatmap(correlation, annot=True)
    plt.title("Correlation Heatmap")
    plt.show()

    # Top category counts
    df["Department"].value_counts().head(10).plot(
        kind="bar",
        title="Top Department Counts"
    )
    plt.show()

    return df


practice()

print("Done! Review with CIA for feedback.")