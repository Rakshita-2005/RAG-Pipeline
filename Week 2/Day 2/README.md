# W2D2 Feature Scaling & Selection

## Scaling Techniques

### StandardScaler
- Converts features into mean 0 and standard deviation 1.
- Sensitive to outliers.

### MinMaxScaler
- Converts values between 0 and 1.
- Good for neural networks.

### RobustScaler
- Uses median and IQR.
- Handles outliers better.


## Feature Selection

SelectKBest with ANOVA F-test was used.

Top 5 features were selected based on their relationship with target variable.


## Encoding

LabelEncoder:
- Converts categories into numbers.

OneHotEncoder:
- Used for unordered categories.

OrdinalEncoder:
- Used when categories have a ranking.


## Viva Preparation

1. OneHotEncoder vs OrdinalEncoder:
- OneHotEncoder → No order categories.
- OrdinalEncoder → Ordered categories.

2. StandardScaler and outliers:
- Outliers affect mean and standard deviation.

3. Feature leakage:
- Happens when information from test/target leaks into training.
- Prevent by splitting data before preprocessing.