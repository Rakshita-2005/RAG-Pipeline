# W1D5: Data Visualisation - Matplotlib & Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def practice():

    # Creating sample dataset
    data = {
        "Student": ["A", "B", "C", "D", "E"],
        "Marks": [85, 70, 90, 60, 75],
        "Study_Hours": [5, 3, 6, 2, 4]
    }

    # Convert data into DataFrame
    
    df = pd.DataFrame(data)

    print("Dataset:")
    print(df)


    # 1. Bar Plot
    plt.figure(figsize=(6,4))
    sns.barplot(
        x="Student",
        y="Marks",
        data=df
    )

    plt.title("Student Marks")
    plt.savefig("marks_barplot.png")
    plt.show()


    # 2. Scatter Plot
    plt.figure(figsize=(6,4))

    sns.scatterplot(
        x="Study_Hours",
        y="Marks",
        data=df
    )

    plt.title("Study Hours vs Marks")
    plt.savefig("study_marks_scatter.png")
    plt.show()


    # 3. Correlation Heatmap
    plt.figure(figsize=(5,4))

    correlation = df.select_dtypes(
        include="number"
    ).corr()

    sns.heatmap(
        correlation,
        annot=True
    )

    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.show()


    return df



# Test
practice()

print("Done! Review with CIA for feedback.")