import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from alpaca.data.requests import StockBarsRequest
import time

client = StockHistoricalDataClient(api_key='PKM8ODVRNO93YD4QCDSZ', secret_key='bal6XoOYjRxtZUcLsBSvYaN4EMjUr5xu6jzSQa7r')

p0 = pd.read_csv('test_panelv2.csv')

p0['initPrice'] = None

def get_open_price(symbol, date):
    try:
        request = StockBarsRequest(symbol_or_symbols=symbol, start=date, timeframe=TimeFrame.Day)
        bars = client.get_stock_bars(request)
        df = bars.df.reset_index(level=None, drop=False, inplace=False)
        df.set_index('timestamp', inplace=True)
        open_price = df.loc[date, 'open']

        return open_price

    except Exception as e:
        print(f"Error fetching data for {symbol} on {date}: {e}")
        return None

for index, row in p0.iterrows():
    symbol = row['issuerTradingSymbol']
    date = f"{row['periodOfReport']} 00:00:00"

    try:
        open_price = get_open_price(symbol, date)
        p0.at[index, 'initPrice'] = open_price
        p0.to_csv('test_panelv2.csv', index=False)
        print(f"Symbol: {symbol}, Date: {date}")
    except Exception as e:
        print(f"Error fetching data for Symbol: {symbol}, Date: {date}: {e}")
    
    time.sleep(1)  # Sleep for 1 second between API calls
