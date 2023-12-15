"""
Author: George Harrison
Date: 10-31-23
Description: script to retrieve stock prices from IEX Cloud API
"""

import pandas as pd
from iexfinance.stocks import get_historical_data
from datetime import datetime, timedelta
import time

iex_token = ''

def get_data(symbols, stock_price_data):
    """
    Gets and saves stock price data for given symbols

    Parameters
    --------------------
        symbols             -- list of stock symbols
        stock_price_data    -- dataframe to append stock price data to

    Returns
    --------------------
        stock_price_data    -- dataframe with stock price data appended
    """

    try:
        json = get_historical_data(symbols=symbols, start='2008-11-01', output_format='json', close_only=True, token=iex_token)
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

stock_price_data = pd.DataFrame()

for df in pd.read_csv('unique_stocks.csv', chunksize=100):
    symbols = df['issuerTradingSymbol'].values.tolist()

    try:
        stock_price_data = get_data(symbols, stock_price_data)
    except Exception as e:
        print(f"Error fetching data for Symbol: {symbols}: {e}")
    time.sleep(10)

pd.set_option('display.max_columns', None)
#stock_price_data.to_csv('stockprices.csv', index=False)




# ##### TEST V1 #####
# stock_list = pd.read_csv('unique_stocks.csv')
# stocks = stock_list['issuerTradingSymbol'].values.tolist()
# stocks = stocks[:100]
# #cost .681 credits
# #goog only goes back to 2014???
# json = get_historical_data(symbols=stocks, start='2008-11-01', output_format='json', close_only=True, token=iex_token)
# df = pd.DataFrame(json)
# df.to_json('stockprice_200.json')
# goog = df['GOOG']
# goog = goog.explode('chart')
# goog = pd.json_normalize(goog)
# goog = goog.drop(columns=['volume', 'change', 'changePercent', 'changeOverTime'])
# goog['issuerTradingSymbol'] = 'GOOG'
# stock_data = pd.concat([], ignore_index=True)



# #### SCRIPT TO RUN MANUALLY ####
# stocks_total = stocks_total[100:] #remove the 100 i just processed
# stocks = stocks_total[:100] #get next 100 to process
# json = get_historical_data(symbols=stocks, start='2008-11-01', output_format='json', close_only=True, token=iex_token)
# df = pd.DataFrame(json)
# df.to_json('stockprice_2000.json')

# for column in df.columns:
#     stock = df[column]
#     stock = stock.explode('chart')
#     stock = pd.json_normalize(stock)
#     stock['issuerTradingSymbol'] = column
#     stock_data = pd.concat([stock_data, stock], ignore_index=True)

# stock_data = stock_data.drop(columns=['volume', 'change', 'changePercent', 'changeOverTime'])
# stock_data.to_csv('stock_prices.csv', index=False)