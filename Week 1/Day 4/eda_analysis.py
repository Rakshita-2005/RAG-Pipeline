# W1D4: Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Create dataset
data = {
    "Age": [21, 25, 30, 35, 40, 45],
    "Salary": [25000, 35000, 50000, 60000, 75000, 90000],
    "Department": ["IT", "HR", "IT", "Finance", "IT", "HR"]
}

# Convert data into DataFrame
df = pd.DataFrame(data)


# 1. Dataset information
print("Dataset Information:")
df.info()


# 2. Statistical summary
print("\nDescription:")
print(df.describe())


# 3. Missing values
print("\nMissing Values:")
print(df.isnull().sum())


# 4. Distribution of numeric columns
df.hist(figsize=(8, 4))
plt.tight_layout()
plt.savefig("distribution_plot.png")
plt.show()


# 5. Correlation heatmap
correlation = df.select_dtypes(include="number").corr()

sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()


# 6. Top category counts
df["Department"].value_counts().head(10).plot(
    kind="bar",
    title="Top Department Counts"
)

plt.tight_layout()
plt.savefig("top_category_counts.png")
plt.show()


# 7. Observations
print("""
EDA Observations:
1. Dataset contains age, salary, and department information.
2. Salary values show variation across employees.
3. IT department has the highest number of records.
4. No missing values are present in the dataset.
5. Correlation analysis helps understand relationships between numerical columns.
""")


print("Done! Review with CIA for feedback.")