"""
Author: George Harrison
Date: 12-05-23
Description: main script recording all data processing done on training dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import anderson
import statsmodels.api as sm
import matplotlib.pyplot as plt

panel = pd.read_csv('../dataverse_files/lit_panel.csv')
                    
#deleting all form types that aren't 4 or 4/A
panel = panel[panel['type'].str.contains('4') | panel['type'].str.contains('4/A')]

#deleting all holding filings and options filings
panel = panel[panel['transactionType'].str.contains('nonDerivativeTransaction')]

#irrelevant columns
panel = panel.drop(columns = 'securityTitleFn')
panel = panel.drop(columns = 'transactionDateFn')
panel = panel.drop(columns = 'deemedExecutionDateFn')
panel = panel.drop(columns = 'transactionCodeFn')
panel = panel.drop(columns = 'transactionTimelinessFn')
panel = panel.drop(columns = 'transactionSharesFn')
panel = panel.drop(columns = 'transactionPricePerShareFn')
panel = panel.drop(columns = 'transactionAcquiredDisposedCdFn')
panel = panel.drop(columns = 'sharesOwnedFolwngTransactionFn')
panel = panel.drop(columns = 'valueOwnedFolwngTransactionFn')
panel = panel.drop(columns = 'directOrIndirectOwnershipFn')
panel = panel.drop(columns = 'natureOfOwnershipFn')
panel = panel.drop(columns = 'conversionOrExercisePriceFn')
panel = panel.drop(columns = 'transactionTotalValueFn')
panel = panel.drop(columns = 'exerciseDateFn')
panel = panel.drop(columns = 'expirationDateFn')
panel = panel.drop(columns = 'underlyingSecurityTitleFn')
panel = panel.drop(columns = 'underlyingSecuritySharesFn')
panel = panel.drop(columns = 'underlyingSecurityValueFn')
panel = panel.drop(columns = 'natureOfOwnership')
panel = panel.drop(columns = 'issuerCIK')
panel = panel.drop(columns = 'tableRow')
panel = panel.drop(columns = 'conversionOrExercisePriceDate')
panel = panel.drop(columns = 'conversionOrExercisePrice')
panel = panel.drop(columns = 'exerciseDate')
panel = panel.drop(columns = 'expirationDate')
panel = panel.drop(columns = 'acceptanceDatetime')
panel = panel.drop(columns = 'underlyingSecurityShares')
panel = panel.drop(columns = 'underlyingSecurityTitle')
panel = panel.drop(columns = 'transactionPricePerShare')
panel = panel.drop(columns = 'deemedExecutionDate')
panel = panel.drop(columns = 'transactionDate')
panel = panel.drop(columns = 'transactionFormType')
panel = panel.drop(columns = 'securityTitle')
panel = panel.drop(columns = 'dateOfOriginalSubmission')
panel = panel.drop(columns = 'dateOfFilingDateChange')
panel = panel.drop(columns = 'filingDate')
panel = panel.drop(columns = 'accessionNumber')
                   
#nearly empty columns
panel = panel.drop(columns = 'valueOwnedFollowingTransaction')
panel = panel.drop(columns = 'transactionTotalValue')
panel = panel.drop(columns = 'underlyingSecurityValue')
                   
#duplicate columns
panel = panel.drop(columns = 'documentType')
panel = panel.drop(columns = 'period')

#drop null rows
panel = panel.dropna(subset=['sharesOwnedFollowingTransaction'])

#drop all rows with dates older than 15 years
panel['periodOfReport'] = pd.to_datetime(panel['periodOfReport'], errors='coerce')
panel = panel[panel['periodOfReport'] > (datetime.now() - timedelta(days=15 * 365))]

#drop all rows with dates less than 1 year old
panel['periodOfReport'] = pd.to_datetime(panel['periodOfReport'], errors='coerce')
panel = panel[panel['periodOfReport'] < (datetime.now() - timedelta(days=367))]

#delete rows with NONE as stock ticker
panel = panel.dropna(subset=['issuerTradingSymbol'])
panel = panel[panel['issuerTradingSymbol'] != 'NONE']

#fix formatting for notSubjectToSection16
#make types consistent
panel['notSubjectToSection16'] = panel['notSubjectToSection16'].replace('0.0', '0')
panel['notSubjectToSection16'] = panel['notSubjectToSection16'].replace('1.0', '1')
panel['notSubjectToSection16'] = panel['notSubjectToSection16'].replace('true', '1')
panel['notSubjectToSection16'] = panel['notSubjectToSection16'].replace('false', '0')

#fill missing values as 0
panel['notSubjectToSection16'] = panel['notSubjectToSection16'].fillna('0')

#convert type of column to integer
panel['notSubjectToSection16'] = panel['notSubjectToSection16'].astype(int)
      
#fill empty rows of transactionTimeliness
#empty means on time
panel['transactionTimeliness'] = panel['transactionTimeliness'].fillna('O')
      
#change types of all columns to be accurate and correct
panel['issuerTradingSymbol'] = panel['issuerTradingSymbol'].astype(str)
panel = panel.dropna(subset=['issuerTradingSymbol'])

panel['equitySwapInvolved'] = panel['equitySwapInvolved'].replace('true', '1')
panel['equitySwapInvolved'] = panel['equitySwapInvolved'].replace('false', '0')

#merge transactionAcquiredDisposedCode with transactionShares
panel.loc[panel['transactionAcquiredDisposedCode'] == 'D', 'transactionShares'] *= -1
panel = panel.drop(columns = 'transactionAcquiredDisposedCode')

##create new columns now that price data has been added
#sharesBeforeTransaction = sharesOwnedFollowingTransaction - transactionShares
panel['sharesBeforeTransaction'] = panel['sharesOwnedFollowingTransaction'] - panel['transactionShares']
#percentVolumeChange = transactionShares / sharesBeforeTransaction
panel['percentVolumeChange'] = panel['transactionShares'] / panel['sharesBeforeTransaction']
panel['percentVolumeChange'] = panel['percentVolumeChange'] * 100
#percentChangeStockPrice = end_prices / init_prices
panel['percentPriceChange'] = (panel['end_prices'] - panel['init_prices']) / panel['init_prices']
panel['percentPriceChange'] = panel['percentPriceChange'] * 100

panel = panel.dropna(subset=['percentVolumeChange'])

#replace inf values with nan
panel = panel.replace([np.inf, -np.inf], np.nan)

#delete all rows with transaction shares = 0
panel = panel[panel['transactionShares'] != 0]

#delete all rows with transactionShares <= |1|
panel = panel[abs(panel['transactionShares']) > 1]

#checking to see that everything went well
print(panel[panel['transactionShares'].between(-1, 1)].shape[0])
print(panel[panel['percentVolumeChange'].between(-1, 1)].shape[0])
print(panel.shape)

## Step 3 - check if data is normally distributed
## method 1 - anderson-darling test
result = anderson(panel['percentPriceChange'])
print(f'Test Statistic: {result.statistic}, Critical Values: {result.critical_values}')

if result.statistic < result.critical_values[2]:
    print('The data appears to be normally distributed.')
else:
    print('The data does not appear to be normally distributed.')

## method 2 - quantile quantile plot
fig = sm.qqplot(panel['percentPriceChange'], line='45')
plt.show(block=True)

## Step 4 - find outliers in dataset
# since data is not normally distributed, use IQR method
# finding iqr
percentile25 = panel['percentPriceChange'].quantile(0.25)
percentile75 = panel['percentPriceChange'].quantile(0.75)
iqr = percentile75 - percentile25
# finding upper and lower limits
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
# finding outliers
panel[panel['percentPriceChange'] > upper_limit]
panel[panel['percentPriceChange'] < lower_limit]
# trim outliers
panel = panel[(panel['percentPriceChange'] <= upper_limit) & (panel['percentPriceChange'] >= lower_limit)]

## removing more outliers
panel = panel[panel['transactionCode'].str.contains('A') | panel['transactionCode'].str.contains('M') 
              | panel['transactionCode'].str.contains('F') | panel['transactionCode'].str.contains('P') 
              | panel['transactionCode'].str.contains('J') | panel['transactionCode'].str.contains('G')
              | panel['transactionCode'].str.contains('C') | panel['transactionCode'].str.contains('D')]

panel = panel[panel['equitySwapInvolved'] == 0]
panel = panel.drop(columns = 'equitySwapInvolved')

## removing percentVolumeChange outliers
percentile25 = panel['percentVolumeChange'].quantile(0.25)
percentile75 = panel['percentVolumeChange'].quantile(0.75)
iqr = percentile75 - percentile25
# finding upper and lower limits
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
# trim outliers
panel = panel[(panel['percentVolumeChange'] <= upper_limit) & (panel['percentVolumeChange'] >= lower_limit)]

## removing transactionShares outliers
percentile25 = panel['transactionShares'].quantile(0.25)
percentile75 = panel['transactionShares'].quantile(0.75)
iqr = percentile75 - percentile25
# finding upper and lower limits
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
# trim outliers
panel = panel[(panel['transactionShares'] <= upper_limit) & (panel['transactionShares'] >= lower_limit)]

## convert transactionTimeliness to binary
panel['transactionTimeliness'] = panel['transactionTimeliness'].replace('E', '0')
panel['transactionTimeliness'] = panel['transactionTimeliness'].replace('O', '1')

## drop type column
panel = panel.drop(columns = 'type')

## convert transactionCode to binary
panel = pd.get_dummies(panel, columns=['transactionCode'], prefix='transactionCode', drop_first=True)
transaction_code_columns = [col for col in panel.columns if 'transactionCode' in col]
panel[transaction_code_columns] = panel[transaction_code_columns].astype(int)

## convert direct or indirect ownership to binary
panel['directOrIndirectOwnership'] = panel['directOrIndirectOwnership'].replace('I', '0')
panel['directOrIndirectOwnership'] = panel['directOrIndirectOwnership'].replace('D', '1')

## train V3, turn top 500 tickers into binary columns
top_stocks = panel['issuerTradingSymbol'].value_counts().head(500).index

for stock in top_stocks:
    panel[f'issuerTradingSymbol_{stock}'] = (panel['issuerTradingSymbol'] == stock).astype(int)

panel = panel.drop(columns=['issuerTradingSymbol'])

## train V4, turn top 1000 tickers into binary columns
top_stocks = panel['issuerTradingSymbol'].value_counts().head(1000).index

for stock in top_stocks:
    panel[f'issuerTradingSymbol_{stock}'] = (panel['issuerTradingSymbol'] == stock).astype(int)

panel = panel.drop(columns=['issuerTradingSymbol'])

panel.to_csv('panel.csv', index=False)