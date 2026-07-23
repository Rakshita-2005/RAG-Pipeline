# W2D1 Feature Engineering & Encoding

## Techniques Implemented

### Encoding
1. LabelEncoder
- Converts categories into numerical labels.
- Trade-off: May create false ordering.

2. OneHotEncoder
- Converts categories into binary columns.
- Best for nominal features.

3. OrdinalEncoder
- Used for ordered categories.
- Maintains ranking information.

## Feature Scaling

### StandardScaler
- Converts data into mean=0 and standard deviation=1.
- Sensitive to outliers.

### MinMaxScaler
- Scales values between 0 and 1.
- Useful for neural networks.

### RobustScaler
- Uses median and IQR.
- Works better with outliers.

## Feature Selection

SelectKBest was used to identify important features.

Top features were selected based on statistical scores.

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn