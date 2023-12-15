"""
Author: George Harrison
Date: 12-06-23
Description: Script to predict stock price changes based on insider trading dataset
"""

import pandas as pd
from iexfinance.stocks import get_historical_data
from datetime import datetime, timedelta
import time
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import numpy as np
from scipy.stats import anderson
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import joblib
from tabulate import tabulate

current_date = datetime.now().strftime('%Y-%m-%d')

iex_token = ''

if not os.path.exists('raw_data'):
    os.makedirs('raw_data')    

def print_predictions(table):
    headers = ["Symbol", "Predicted % Change", "Current $", "Predicted $"]
    floatfmt = ("", ".2f", ".2f", ".2f")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=floatfmt))

def predict(panel):
    """
    Prints predictions for top 3 and bottom 3 stocks

    Parameters
    --------------------
        panel               -- fully cleaned dataset to predict on
    """

    os.system('cls' if os.name=='nt' else 'clear')
    print('now predicting')
    columns_raw = pd.read_csv('column_names.csv')
    columns = [column for sublist in columns_raw.values.tolist() for column in sublist]

    X = panel.drop(columns = ['issuerTradingSymbol', 'periodOfReport'])
    # make sure columns are consistent with columns in trained dataset
    X = X[columns]

    # saving stock reference list so we can see what stocks are recommended later
    stock_ref = pd.DataFrame()
    stock_ref['issuerTradingSymbol'] = panel['issuerTradingSymbol']
    stock_ref['init_prices'] = panel['init_prices']
    stock_ref['periodOfReport'] = panel['periodOfReport']

    RidgeModel = joblib.load('trained_models/Ridge_1000tickers.joblib')
    predictions = RidgeModel.predict(X)
    stock_ref['percentPriceChange'] = predictions
    #sort out duplicates
    stock_ref = stock_ref.loc[stock_ref.groupby('issuerTradingSymbol')['periodOfReport'].idxmax()]
    stock_ref = stock_ref.sort_values(by='percentPriceChange', ascending=False)
    stock_ref['predictedPrice'] = stock_ref['init_prices'] * (1 + (stock_ref['percentPriceChange'] / 100))

    pos_pred = []
    neg_pred = []
    for index, row in stock_ref.head(3).iterrows():
        pos_pred.append((row['issuerTradingSymbol'], row['percentPriceChange'], row['init_prices'], row['predictedPrice']))
    for index, row in stock_ref.tail(3).iterrows():
        neg_pred.append((row['issuerTradingSymbol'], row['percentPriceChange'], row['init_prices'], row['predictedPrice']))

    print(tabulate([[f'--------- [StockBuddy {current_date} TOP PREDICTIONS] ---------']], tablefmt="fancy_grid"))
    print_predictions(pos_pred)
    print(tabulate([[f'-------- [StockBuddy {current_date} BOTTOM PREDICTIONS] --------']], tablefmt="fancy_grid"))
    print_predictions(neg_pred[::-1])

    #save predictions to txt file
    with open(f'raw_data/predictions_{current_date}.txt', 'w') as file:
        file.write(tabulate([[f'--------- [StockBuddy {current_date} TOP PREDICTIONS] ---------']], tablefmt="fancy_grid") + '\n')

        pos_pred = []
        for index, row in stock_ref.head(3).iterrows():
            pos_pred.append((row['issuerTradingSymbol'], row['percentPriceChange'], row['init_prices'], row['predictedPrice']))

        headers = ["Symbol", "Predicted % Change", "Current $", "Predicted $"]
        floatfmt = ("", ".2f", ".2f", ".2f")

        file.write(tabulate(pos_pred, headers=headers, tablefmt="fancy_grid", floatfmt=floatfmt))

        file.write(tabulate([[f'-------- [StockBuddy {current_date} BOTTOM PREDICTIONS] --------']], tablefmt="fancy_grid") + '\n')

        neg_pred = []
        for index, row in stock_ref.tail(3).iterrows():
            neg_pred.append((row['issuerTradingSymbol'], row['percentPriceChange'], row['init_prices'], row['predictedPrice']))

        file.write(tabulate(neg_pred[::-1], headers=headers, tablefmt="fancy_grid", floatfmt=floatfmt))


def get_data(symbols, stock_price_data):
    try:
        json = get_historical_data(symbols=symbols, start='2022-12-06', output_format='json', close_only=True, token=iex_token)
        df = pd.DataFrame(json)

        for column in df.columns:
            stock = df[column]
            stock = stock.explode('chart')
            stock = pd.json_normalize(stock)
            stock['issuerTradingSymbol'] = column
            stock_price_data = pd.concat([stock_price_data, stock], ignore_index=True)

        stock_price_data = stock_price_data.drop(columns=['volume', 'change', 'changePercent', 'changeOverTime'])

        return stock_price_data

    except Exception as e:
        print(f"Error fetching data for {symbols}: {e}")
        return None
    
def get_prices():
    print('now getting price data')
    stock_price_data = pd.DataFrame()

    panel = pd.read_csv(f'raw_data/panel_step1_{current_date}.csv')

    symbols = panel['issuerTradingSymbol'].drop_duplicates().head(500)
    symbols.to_csv(f'raw_data/panel_symbols_{current_date}.csv', index=False)

    for df in tqdm(pd.read_csv(f'raw_data/panel_symbols_{current_date}.csv', chunksize=100)):
        try:
            symbols = [symbol for sublist in df.values.tolist() for symbol in sublist]
            stock_price_data = get_data(symbols, stock_price_data)

        except Exception as e:
            print(f"Error fetching data for Symbol: {symbols}: {e}")
        time.sleep(10)

    stock_price_data.to_csv(f'raw_data/panel_stockprices_{current_date}.csv', index=False)
    add_prices(stock_price_data)

def add_prices(stock_prices):
    print('now adding price data')
    stock_prices = stock_prices.dropna()

    stock_data = {}
    for _, row in tqdm(stock_prices.iterrows()):
        symbol = row['issuerTradingSymbol']
        date = row['date']
        close = row['close']
        if symbol not in stock_data:
            stock_data[symbol] = {}
        stock_data[symbol][date] = close

    result_df = pd.DataFrame()
    chunksize = 10000

    for df in pd.read_csv(f'raw_data/panel_step1_{current_date}.csv', chunksize=chunksize):
        df['init_prices'] = None

        for _, row in tqdm(df.iterrows()):
            start_date = str(row['periodOfReport'])
            symbol = row['issuerTradingSymbol']
            if symbol in stock_data and start_date in stock_data[symbol]:
                df.loc[_, 'init_prices'] = stock_data[symbol][start_date]
        result_df = pd.concat([result_df, df])
    print(result_df.shape)
    result_df = result_df.dropna(subset='init_prices')
    print(result_df.shape)
    result_df.to_csv(f'raw_data/panel_step2_{current_date}.csv', index=False)
    clean_price_data(result_df)

def clean_raw_data(panel):
    print('now cleaning raw data')
    panel = panel.dropna(subset=['transactionType'])
    panel = panel[panel['type'].str.contains('4') | panel['type'].str.contains('4/A')]
    panel = panel[panel['transactionType'].str.contains('nonDerivativeTransaction')]
    panel = panel.drop(columns = ['securityTitleFn', 'transactionDateFn', 'deemedExecutionDateFn', 'transactionCodeFn', 'transactionTimelinessFn', 
                                    'transactionSharesFn', 'transactionPricePerShareFn', 'transactionAcquiredDisposedCdFn', 'sharesOwnedFolwngTransactionFn', 
                                    'valueOwnedFolwngTransactionFn', 'directOrIndirectOwnershipFn', 'natureOfOwnershipFn', 'conversionOrExercisePriceFn', 
                                    'transactionTotalValueFn', 'exerciseDateFn', 'expirationDateFn', 'underlyingSecurityTitleFn', 'underlyingSecuritySharesFn', 
                                    'underlyingSecurityValueFn', 'natureOfOwnership', 'issuerCIK', 'tableRow', 'conversionOrExercisePrice',
                                    'exerciseDate', 'expirationDate', 'acceptanceDatetime', 'underlyingSecurityShares', 'underlyingSecurityTitle', 'transactionPricePerShare',
                                    'deemedExecutionDate', 'transactionDate', 'transactionFormType', 'securityTitle', 'dateOfOriginalSubmission', 'dateOfFilingDateChange',
                                    'filingDate', 'accessionNumber', 'valueOwnedFollowingTransaction', 'transactionTotalValue', 'underlyingSecurityValue', 'documentType', 'period'])

    panel = panel.dropna(subset=['sharesOwnedFollowingTransaction'])

    #keep only data from within the past year
    panel['periodOfReport'] = pd.to_datetime(panel['periodOfReport'], errors='coerce')
    panel = panel[panel['periodOfReport'] > (datetime.now() - timedelta(days=365))]

    panel = panel.dropna(subset=['issuerTradingSymbol'])
    panel = panel[panel['issuerTradingSymbol'] != 'NONE']
    panel['notSubjectToSection16'] = panel['notSubjectToSection16'].replace('0.0', '0')
    panel['notSubjectToSection16'] = panel['notSubjectToSection16'].replace('1.0', '1')
    panel['notSubjectToSection16'] = panel['notSubjectToSection16'].replace('true', '1')
    panel['notSubjectToSection16'] = panel['notSubjectToSection16'].replace('false', '0')
    panel['notSubjectToSection16'] = panel['notSubjectToSection16'].fillna('0')
    panel['notSubjectToSection16'] = panel['notSubjectToSection16'].astype(int)
    panel['transactionTimeliness'] = panel['transactionTimeliness'].fillna('O')
    panel['issuerTradingSymbol'] = panel['issuerTradingSymbol'].astype(str)
    panel = panel.dropna(subset=['issuerTradingSymbol'])
    panel['equitySwapInvolved'] = panel['equitySwapInvolved'].replace('true', '1')
    panel['equitySwapInvolved'] = panel['equitySwapInvolved'].replace('false', '0')
    panel.loc[panel['transactionAcquiredDisposedCode'] == 'D', 'transactionShares'] *= -1
    panel = panel.drop(columns = 'transactionAcquiredDisposedCode')

    panel['sharesOwnedFollowingTransaction'] = pd.to_numeric(panel['sharesOwnedFollowingTransaction'], errors='coerce')
    panel['transactionShares'] = pd.to_numeric(panel['transactionShares'], errors='coerce')
    panel['sharesBeforeTransaction'] = panel['sharesOwnedFollowingTransaction'] - panel['transactionShares']

    panel['sharesBeforeTransaction'] = pd.to_numeric(panel['sharesBeforeTransaction'], errors='coerce')
    panel['percentVolumeChange'] = panel['transactionShares'] / panel['sharesBeforeTransaction']
    panel['percentVolumeChange'] = panel['percentVolumeChange'] * 100

    panel.to_csv(f'raw_data/panel_step1_{current_date}.csv', index=False)
    get_prices()

def clean_price_data(panel):
    print('now cleaning price data')
    panel = panel.dropna(subset=['percentVolumeChange'])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel[panel['transactionShares'] != 0]
    panel = panel[abs(panel['transactionShares']) > 1]
    panel = panel.drop(columns = 'equitySwapInvolved')
    percentile25 = panel['percentVolumeChange'].quantile(0.25)
    percentile75 = panel['percentVolumeChange'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    panel = panel[(panel['percentVolumeChange'] <= upper_limit) & (panel['percentVolumeChange'] >= lower_limit)]
    percentile25 = panel['transactionShares'].quantile(0.25)
    percentile75 = panel['transactionShares'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    panel = panel[(panel['transactionShares'] <= upper_limit) & (panel['transactionShares'] >= lower_limit)]
    panel = panel[panel['transactionCode'].str.contains('A') | panel['transactionCode'].str.contains('M') 
              | panel['transactionCode'].str.contains('F') | panel['transactionCode'].str.contains('P') 
              | panel['transactionCode'].str.contains('J') | panel['transactionCode'].str.contains('G')
              | panel['transactionCode'].str.contains('C') | panel['transactionCode'].str.contains('D')]
    panel['transactionTimeliness'] = panel['transactionTimeliness'].replace('E', '0')
    panel['transactionTimeliness'] = panel['transactionTimeliness'].replace('O', '1')
    panel = panel.drop(columns = ['type', 'transactionType'])
    panel = pd.get_dummies(panel, columns=['transactionCode'], prefix='transactionCode', drop_first=True)
    transaction_code_columns = [col for col in panel.columns if 'transactionCode' in col]
    panel[transaction_code_columns] = panel[transaction_code_columns].astype(int)
    panel['directOrIndirectOwnership'] = panel['directOrIndirectOwnership'].replace('I', '0')
    panel['directOrIndirectOwnership'] = panel['directOrIndirectOwnership'].replace('D', '1')

    ## convert 1000 stocks to binary, try new method that avoids performance warning
    top_stocks = pd.read_csv('top_1000_stocks.csv')
    stocks = [symbol for sublist in top_stocks.values.tolist() for symbol in sublist]

    for stock in stocks:
        panel[f'issuerTradingSymbol_{stock}'] = (panel['issuerTradingSymbol'] == stock).astype(int)

    panel = panel.drop(columns = 'URL')
    panel = panel.drop(columns = 'sharesBeforeTransaction')

    panel.to_csv(f'raw_data/panel_step3_{current_date}.csv', index=False)
    predict(panel)


def download_data():
    print(f'downloading raw data from kaggle for {current_date}')
    api = KaggleApi()
    api.authenticate() 
    api.dataset_download_cli('layline/insidertrading', file_name='lit_panel.csv', path='raw_data/')
    unzip_data()

def unzip_data():
    print('unzipping raw data')
    with zipfile.ZipFile('raw_data/lit_panel.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('raw_data/')
    panel = pd.read_csv('raw_data/lit_panel.csv')
    clean_raw_data(panel)

if os.path.exists(f'raw_data/panel_step3_{current_date}.csv'):
    panel = pd.read_csv(f'raw_data/panel_step3_{current_date}.csv')
    predict(panel)
elif os.path.exists(f'raw_data/panel_step2_{current_date}.csv'):
    panel = pd.read_csv(f'raw_data/panel_step2_{current_date}.csv')
    clean_price_data(panel)
elif os.path.exists(f'raw_data/panel_stockprices_{current_date}.csv'):
    stock_prices = pd.read_csv(f'raw_data/panel_stockprices_{current_date}.csv')
    add_prices(stock_prices)
elif os.path.exists(f'raw_data/panel_step1_{current_date}.csv'):
    panel = pd.read_csv(f'raw_data/panel_step1_{current_date}.csv')
    get_prices()
elif os.path.exists('raw_data/lit_panel.csv'):
    panel = pd.read_csv('raw_data/lit_panel.csv')
    clean_raw_data(panel)
elif os.path.exists('raw_data/lit_panel.csv.zip'):
    unzip_data()
else:
    download_data()