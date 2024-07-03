import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Scaling 'Amount' with RobustScaler to handle outliers
scaler_amount = RobustScaler()
data['Amount'] = scaler_amount.fit_transform(data[['Amount']])

# Standardizing 'Time'
scaler_time = StandardScaler()
data['Time'] = scaler_time.fit_transform(data[['Time']])

# Save preprocessed data to a CSV file
data.to_csv('preprocessed_creditcard.csv', index=False)

# Save scaled feature values to a CSV file
scaled_features = pd.DataFrame({'Amount': data['Amount'], 'Time': data['Time']})
scaled_features.to_csv('scaled_features.csv', index=False)
