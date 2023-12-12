"""
Author: George Harrison
Date: 11-02-23
Description: appends stock prices which have already been retrieved from API to the dataset
"""

import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

stock_prices = pd.read_csv('stock_prices.csv')
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

for df in pd.read_csv('panel_cleaned.csv', chunksize=chunksize):
    df['init_prices'] = None
    df['end_dates'] = None
    df['end_prices'] = None

    for _, row in tqdm(df.iterrows()):
        start_date = str(row['periodOfReport'])
        symbol = row['issuerTradingSymbol']

        if symbol in stock_data and start_date in stock_data[symbol]:
            df.at[_, 'init_prices'] = stock_data[symbol][start_date]

            max_days_to_search = 30
            end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=365)
            while max_days_to_search > 0:
                end_date_str = end_date.strftime('%Y-%m-%d')
                if symbol in stock_data and end_date_str in stock_data[symbol]:
                    df.at[_, 'end_dates'] = pd.to_datetime(end_date_str)
                    df.at[_, 'end_prices'] = stock_data[symbol][end_date_str]
                    break
                else:
                    end_date += timedelta(days=1)
                    max_days_to_search -= 1

    result_df = pd.concat([result_df, df])

result_df.to_csv('panel_cleaned_adddates.csv', index=False)
