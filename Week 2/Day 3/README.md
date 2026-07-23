# W2D3 Handling Imbalanced Data - SMOTE

## Objective
Handle class imbalance using SMOTE technique.

## What I Implemented

- Created imbalanced classification dataset
- Checked class distribution
- Applied SMOTE oversampling
- Trained Logistic Regression model
- Evaluated model performance


## Why SMOTE?

SMOTE creates synthetic samples for minority classes instead of simply duplicating existing data.


## Design Decisions

- Used SMOTE only on training data to avoid data leakage.
- Used Logistic Regression as a baseline model.


## Viva Answers

### What did you build?
Implemented a pipeline to balance imbalanced datasets using SMOTE.

### Hardest part?
Understanding where to apply SMOTE. Solved by applying it only after train-test split.

### One improvement?
Try advanced models like Random Forest, XGBoost, or tune SMOTE parameters.