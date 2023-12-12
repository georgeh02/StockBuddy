"""
Author: George Harrison
Date: 12-03-23
Description: used to check for multicollinearity and redundancy in the dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', None)

df = pd.read_csv('panel_step12_trainV2.csv')

df = df.drop(columns = 'URL')
df = df.drop(columns = 'issuerTradingSymbol')

# dropping due to multicollinearity and redundancy
df = df.drop(columns = 'end_prices')
df = df.drop(columns = 'sharesBeforeTransaction')

df = df.drop(columns = 'periodOfReport')
df = df.drop(columns = 'end_dates')

X = df.drop('percentPriceChange', axis=1)
y = df['percentPriceChange']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Calculate the correlation matrix
correlation_matrix = X_train.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF
print("\nVIF:")
print(vif_data)

# Eigenvalues of the correlation matrix
eigenvalues, _ = np.linalg.eig(correlation_matrix)

# Display eigenvalues
print("\nEigenvalues of the Correlation Matrix:")
print(eigenvalues)