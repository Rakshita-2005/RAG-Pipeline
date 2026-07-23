# ============================================
# Week 1 Day 2 - Pandas for Data Manipulation
# Name: Rakshita Kurahatti
# ============================================

import pandas as pd
import os

# --------------------------------------------
# Load the CSV file into a Pandas DataFrame
# --------------------------------------------
df = pd.read_csv("indian_states.csv")

# --------------------------------------------
# Print the shape of the DataFrame
# --------------------------------------------
print("========== DATAFRAME SHAPE ==========")
print(df.shape)

# --------------------------------------------
# Print the data types of each column
# --------------------------------------------
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# --------------------------------------------
# Print the first 10 rows
# --------------------------------------------
print("\n========== FIRST 10 ROWS ==========")
print(df.head(10))

# --------------------------------------------
# FILTER
# Filter states where Literacy is greater than 80
# --------------------------------------------
filtered_df = df[df["Literacy"] > 80]

print("\n========== FILTERED DATA (Literacy > 80) ==========")
print(filtered_df)

# --------------------------------------------
# GROUPBY
# Find average population by Region
# --------------------------------------------
grouped_df = df.groupby("Region")["Population"].mean()

print("\n========== GROUPBY (Average Population by Region) ==========")
print(grouped_df)

# --------------------------------------------
# Create another DataFrame for Merge
# --------------------------------------------
capital_df = pd.DataFrame({
    "State": [
        "Karnataka",
        "Tamil Nadu",
        "Kerala",
        "Maharashtra",
        "Gujarat",
        "Rajasthan",
        "Punjab",
        "Uttar Pradesh",
        "West Bengal",
        "Assam",
        "Odisha",
        "Bihar"
    ],
    "Capital": [
        "Bengaluru",
        "Chennai",
        "Thiruvananthapuram",
        "Mumbai",
        "Gandhinagar",
        "Jaipur",
        "Chandigarh",
        "Lucknow",
        "Kolkata",
        "Dispur",
        "Bhubaneswar",
        "Patna"
    ]
})

# --------------------------------------------
# MERGE
# Merge state details with capital cities
# --------------------------------------------
merged_df = pd.merge(df, capital_df, on="State", how="inner")

print("\n========== MERGED DATA ==========")
print(merged_df)

# --------------------------------------------
# PIVOT TABLE
# Total population by Region
# --------------------------------------------
pivot_table = pd.pivot_table(
    df,
    values="Population",
    index="Region",
    aggfunc="sum"
)

print("\n========== PIVOT TABLE ==========")
print(pivot_table)

# --------------------------------------------
# Clean Data
# Remove duplicate rows
# --------------------------------------------
cleaned_df = df.drop_duplicates()

# --------------------------------------------
# Export cleaned data to CSV
# --------------------------------------------
cleaned_df.to_csv("cleaned_data.csv", index=False)

# --------------------------------------------
# Export cleaned data to Parquet
# --------------------------------------------
cleaned_df.to_parquet("cleaned_data.parquet", index=False)

# --------------------------------------------
# Compare file sizes
# --------------------------------------------
csv_size = os.path.getsize("cleaned_data.csv")
parquet_size = os.path.getsize("cleaned_data.parquet")

print("\n========== FILE SIZE COMPARISON ==========")
print(f"CSV File Size     : {csv_size} bytes")
print(f"Parquet File Size : {parquet_size} bytes")

if parquet_size < csv_size:
    print("\nParquet file is smaller and more storage-efficient.")
elif parquet_size > csv_size:
    print("\nCSV file is smaller.")
else:
    print("\nBoth files have the same size.")

print("\n==========================================")
print("Assignment Completed Successfully!")
print("==========================================")