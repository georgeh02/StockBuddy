"""
Author: George Harrison
Date: 12-06-23
Description: script recording all data visualization on training results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
import joblib

df = pd.read_csv('panel_step14_1000stocks.csv')

## PREPROCESSING--------------------------------------------------------
df = df.drop(columns = 'URL')
df = df.drop(columns = 'end_prices') # this column will be blank in my real tests

# dropping due to multicollinearity and redundancy
df = df.drop(columns = 'sharesBeforeTransaction')

# dates
df = df.drop(columns = 'periodOfReport')
df = df.drop(columns = 'end_dates')


X = df.drop('percentPriceChange', axis=1)
y = df['percentPriceChange']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Ridge = joblib.load('trained_models/Ridge_1000tickers.joblib')
SGD = joblib.load('trained_models/SGDRegressor_1000tickers.joblib')
EN = joblib.load('trained_models/ElasticNet_1000tickers.joblib')
Ridge_Pred = Ridge.predict(X_test_scaled)
SGD_Pred = SGD.predict(X_test_scaled)
EN_Pred = EN.predict(X_test_scaled)


# # HISTOGRAMS OF PERCENT PRICE DISTRIBUTION
plt.figure(figsize=(10, 6))
sns.histplot(Ridge_Pred, kde=False, bins=1500)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.title('Histogram - Ridge Predicted percentPriceChange distribution (Log Scale)')
plt.grid(True)
plt.savefig('hist1.png', dpi=300)

plt.figure(figsize=(10, 6))
sns.histplot(SGD_Pred, kde=False, bins=1500)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.title('Histogram - SGD Predicted percentPriceChange distribution (Log Scale)')
plt.grid(True)
plt.savefig('hist2.png', dpi=300)

plt.figure(figsize=(10, 6))
sns.histplot(EN_Pred, kde=False, bins=1500)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.title('Histogram - ElasticNet Predicted percentPriceChange distribution (Log Scale)')
plt.grid(True)
plt.savefig('hist3.png', dpi=300)

# SCATTERPLOTS OF PREDICTED VS ACTUAL PERCENTPRICECHANGE
plt.figure(figsize=(10, 6))
plt.scatter(Ridge_Pred, y_test, alpha=0.1)
plt.title('Ridge percentPriceChange vs. Actual percentPriceChange')
plt.xlabel('percentPriceChange (ridge)')
plt.ylabel('percentPriceChange (actual)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='perfect prediction line')
plt.grid(True)
plt.savefig('scatter1.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.scatter(SGD_Pred, y_test, alpha=0.1)
plt.title('SGD percentPriceChange vs. Actual percentPriceChange')
plt.xlabel('percentPriceChange (SGD)')
plt.ylabel('percentPriceChange (actual)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='perfect prediction line')
plt.grid(True)
plt.savefig('scatter2.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.scatter(EN_Pred, y_test, alpha=0.1)
plt.title('ElasticNet percentPriceChange vs. Actual percentPriceChange')
plt.xlabel('percentPriceChange (ElasicNet)')
plt.ylabel('percentPriceChange (actual)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='perfect prediction line')
plt.grid(True)
plt.savefig('scatter3.png', dpi=300)



# # SCATTERPLOTS OF ACTUAL VS PREDICTED END PRICES V1
init_prices = X_test['init_prices']
end_prices_actual = X_test['init_prices'] * (1 + (y_test / 100))
end_ridge = X_test['init_prices'] * (1 + (Ridge_Pred / 100))
end_sgd = X_test['init_prices'] * (1 + (SGD_Pred / 100))
end_en = X_test['init_prices'] * (1 + (EN_Pred / 100))

plt.figure(figsize=(10, 6))
plt.scatter(init_prices, end_prices_actual, label='Actual', marker='o', color='blue')
for x, y1, y2 in zip(init_prices, end_prices_actual, end_prices_actual):
    plt.plot([x, x], [y1, y2], color='blue')
plt.scatter(init_prices, end_ridge, label='Predicted', marker='x', color='red')
for x, y1, y2 in zip(init_prices, end_ridge, end_ridge):
    plt.plot([x, x], [y1, y2], color='red', linestyle='--')

plt.xlabel('Initial Price')
plt.ylabel('End Price')
plt.title('Actual vs Predicted End Prices (Ridge)')
plt.legend()
plt.grid(True)
plt.savefig('scatter4.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.scatter(init_prices, end_prices_actual, label='Actual', marker='o', color='blue')
for x, y1, y2 in zip(init_prices, end_prices_actual, end_prices_actual):
    plt.plot([x, x], [y1, y2], color='blue')
plt.scatter(init_prices, end_sgd, label='Predicted', marker='x', color='red')
for x, y1, y2 in zip(init_prices, end_sgd, end_sgd):
    plt.plot([x, x], [y1, y2], color='red', linestyle='--')

plt.xlabel('Initial Price')
plt.ylabel('End Price')
plt.title('Actual vs Predicted End Prices (SGD)')
plt.legend()
plt.grid(True)
plt.savefig('scatter5.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.scatter(init_prices, end_prices_actual, label='Actual', marker='o', color='blue')
for x, y1, y2 in zip(init_prices, end_prices_actual, end_prices_actual):
    plt.plot([x, x], [y1, y2], color='blue')
plt.scatter(init_prices, end_en, label='Predicted', marker='x', color='red')
for x, y1, y2 in zip(init_prices, end_en, end_en):
    plt.plot([x, x], [y1, y2], color='red', linestyle='--')

plt.xlabel('Initial Price')
plt.ylabel('End Price')
plt.title('Actual vs Predicted End Prices (ElasticNet)')
plt.legend()
plt.grid(True)
plt.savefig('scatter6.png', dpi=300)


# # GROUPED BAR CHART DIRECTIONAL ACCURACY
y_test_reset = y_test.reset_index(drop=True)
results_ridge = pd.DataFrame()
results_ridge['Predicted_Positive'] = Ridge_Pred > 0
results_ridge['Actual_Positive'] = y_test_reset > 0

results_sgd = pd.DataFrame()
results_sgd['Predicted_Positive'] = SGD_Pred > 0
results_sgd['Actual_Positive'] = y_test_reset > 0

results_en = pd.DataFrame()
results_en['Predicted_Positive'] = EN_Pred > 0
results_en['Actual_Positive'] = y_test_reset > 0

grouped_data_ridge = results_ridge.groupby(['Predicted_Positive', 'Actual_Positive']).size().reset_index(name='Ridge_count')
grouped_data_sgd = results_sgd.groupby(['Predicted_Positive', 'Actual_Positive']).size().reset_index(name='SGD_count')
grouped_data_en = results_en.groupby(['Predicted_Positive', 'Actual_Positive']).size().reset_index(name='EN_count')

grouped_data = pd.merge(grouped_data_ridge, grouped_data_sgd, on=['Predicted_Positive', 'Actual_Positive'], how='outer')
grouped_data = pd.merge(grouped_data, grouped_data_en, on=['Predicted_Positive', 'Actual_Positive'], how='outer')

labels = ['Predicted Neg | Actual Neg', 'Predicted Neg | Actual Pos', 'Predicted Pos | Actual Neg', 'Predicted Pos | Actual Pos']
counts_ridge = grouped_data['Ridge_count'].fillna(0)
counts_sgd = grouped_data['SGD_count'].fillna(0)
counts_en = grouped_data['EN_count'].fillna(0)
fig, ax = plt.subplots()
bar_width = 0.2
index = np.arange(len(labels))
bar1 = ax.bar(index - bar_width, counts_ridge, bar_width, label='Ridge')
bar2 = ax.bar(index, counts_sgd, bar_width, label='SGD')
bar3 = ax.bar(index + bar_width, counts_en, bar_width, label='ElasticNet')
ax.set_ylabel('Count')
ax.set_title('Actual Direction vs Predicted Direction')
ax.set_xticks(index)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('barplot1.png', dpi=300)


## CHECKING DISTRIBUTION OF POS AND NEG VLAUES IN ACTUAL
actual_values = y_test_reset > 0
value_counts = actual_values.value_counts()
plt.bar(value_counts.index, value_counts.values, color=['blue', 'red'])
plt.title('Distribution of Actual Positive and Negative Values')
plt.xlabel('Actual Value')
plt.ylabel('Count')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.savefig('barplot2.png', dpi=300)


## CONFUSIONG MATRIX FOR POS AND NEG VALUES
conf_matrix = confusion_matrix(results_ridge['Actual_Positive'], results_ridge['Predicted_Positive'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Neg', 'Predicted Pos'],
            yticklabels=['Actual Neg', 'Actual Pos'])
plt.title('Confusion Matrix (Ridge)')
plt.xlabel('Predicted (Ridge)')
plt.ylabel('Actual')
plt.savefig('confusion_matrix1.png', dpi=300)

conf_matrix = confusion_matrix(results_sgd['Actual_Positive'], results_sgd['Predicted_Positive'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Neg', 'Predicted Pos'],
            yticklabels=['Actual Neg', 'Actual Pos'])
plt.title('Confusion Matrix (SGD)')
plt.xlabel('Predicted (SGD)')
plt.ylabel('Actual')
plt.savefig('confusion_matrix2.png', dpi=300)

conf_matrix = confusion_matrix(results_en['Actual_Positive'], results_en['Predicted_Positive'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Neg', 'Predicted Pos'],
            yticklabels=['Actual Neg', 'Actual Pos'])
plt.title('Confusion Matrix (EN)')
plt.xlabel('Predicted (EN)')
plt.ylabel('Actual')
plt.savefig('confusion_matrix3.png', dpi=300)


# # # LINEPLOT - ACTUAL VS PREDICT PRICE (WORK IN PROGRESS)
# X_test['end_prices_actual'] = X_test['init_prices'] * (1 + (y_test / 100))
# X_test['end_prices_predicted'] = X_test['init_prices'] * (1 + (Ridge_Pred / 100))

# sampled_df = X_test.sample(n=100, random_state=42)

# sampled_df['init_time'] = 0
# sampled_df['real_end_time'] = 1
# sampled_df['predicted_end_time'] = 1

# # Plotting lines between init prices and actual end prices
# plt.plot([sampled_df['init_time'], sampled_df['real_end_time']],
#          [sampled_df['init_prices'], sampled_df['end_prices_actual']], color='blue', label='Actual', alpha=0.5)

# # Plotting lines between init prices and predicted end prices
# plt.plot([sampled_df['init_time'], sampled_df['predicted_end_time']],
#          [sampled_df['init_prices'], sampled_df['end_prices_predicted']], color='red', label='Predicted', alpha=0.5)

# # Add labels and legend
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title('Actual vs Predicted End Prices (Ridge)')
# plt.grid(True)
# plt.savefig('lineplot1.png', dpi=300)