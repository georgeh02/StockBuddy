import pandas as pd
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
import time

ts = TimeSeries(key='50Y9BA7Z4HGJ5GJR', output_format='pandas')

p0 = pd.read_csv('test_panelv2.csv')

p0['initPrice'] = None

def get_open_price(symbol, date):
    try:
        data, meta_data = ts.get_daily(symbol, outputsize='full')
        open_price = data.loc[date, '1. open']

        return open_price

    except Exception as e:
        print(f"Error fetching data for {symbol} on {date}: {e}")
        return None

for index, row in p0.iterrows():
    symbol = row['issuerTradingSymbol']
    date = row['periodOfReport']

    try:
        open_price = get_open_price(symbol, date)
        p0.at[index, 'initPrice'] = open_price
        p0.to_csv('test_panelv2.csv', index=False)
        print(f"Symbol: {symbol}, Date: {date}")
    except Exception as e:
        print(f"Error fetching data for Symbol: {symbol}, Date: {date}: {e}")
    
    time.sleep(1)  # Sleep for 1 second between API calls
