# Week 2 Day 5 - End-to-End Preprocessing Pipeline

## Objective
Implemented an end-to-end ML preprocessing pipeline using Scikit-learn.

## Concepts Implemented

- Handling missing values
- Numerical feature scaling using StandardScaler
- Categorical encoding using OneHotEncoder
- Train-test split
- ColumnTransformer
- Pipeline automation

## Dataset Features

Numerical:
- Age
- Salary

Categorical:
- City

Target:
- Purchased

## Key Design Decisions

1. Used Pipeline to avoid data leakage.
2. Applied fit only on training data.
3. Used ColumnTransformer for different preprocessing strategies.
4. Used OneHotEncoder for categorical variables.

## Testing

Tested execution with multiple dataset inputs.

## Output Evidence

Added execution screenshots in outputs folder.

## Tools Used

- Python
- Pandas
- Scikit-learn
- ML Pipeline concepts