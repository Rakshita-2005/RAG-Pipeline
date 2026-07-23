# W1D3: Data Loading, Cleaning & Inspection

import pandas as pd


def practice():
    # Sample dataset
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Age": [25, None, 30, 28],
        "Score": [85, 90, None, 88]
    }

    # Load data into DataFrame
    df = pd.DataFrame(data)

    print("Original Data:")
    print(df)

    # Inspect dataset
    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Cleaning missing values
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Score"] = df["Score"].fillna(df["Score"].mean())

    print("\nCleaned Data:")
    print(df)

    return df


# Test
practice()
print("Done! Review with CIA for feedback.")