# Week 1 Day 1 - NumPy Fundamentals
import numpy as np
import pandas as pd

print("=" * 50)
print("NUMPY FUNDAMENTALS")
print("=" * 50)

# -----------------------------
# 1D Array
# -----------------------------
arr1 = np.array([10, 20, 30, 40, 50])

print("\n1D Array")
print(arr1)
print("Shape:", arr1.shape)

# -----------------------------
# 2D Array
# -----------------------------
arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\n2D Array")
print(arr2)
print("Shape:", arr2.shape)

# -----------------------------
# 3D Array
# -----------------------------
arr3 = np.array([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
])

print("\n3D Array")
print(arr3)
print("Shape:", arr3.shape)

# -----------------------------
# Broadcasting
# -----------------------------
print("\nBroadcasting Example")

A = np.array([[1],[2],[3]])
B = np.array([10,20,30])

print(A + B)

# -----------------------------
# Vectorized Operations
# -----------------------------
print("\nVectorized Operations")

numbers = np.array([1,2,3,4,5])

print("Original:", numbers)
print("Add 5:", numbers + 5)
print("Multiply by 10:", numbers * 10)
print("Square:", numbers ** 2)

# -----------------------------
# Matrix Multiplication
# -----------------------------
print("\nMatrix Multiplication")

matrix1 = np.array([
    [1,2],
    [3,4]
])

matrix2 = np.array([
    [5,6],
    [7,8]
])

result = np.matmul(matrix1, matrix2)

print(result)

# -----------------------------
# Read CSV Dataset
# -----------------------------
print("\nCSV Dataset")

df = pd.read_csv("student_scores.csv")

print(df)

data = df.to_numpy()

# -----------------------------
# Mean
# -----------------------------
print("\nMean")

print(np.mean(data, axis=0))

# -----------------------------
# Standard Deviation
# -----------------------------
print("\nStandard Deviation")

print(np.std(data, axis=0))

# -----------------------------
# Correlation
# -----------------------------
print("\nCorrelation Matrix")

print(np.corrcoef(data, rowvar=False))