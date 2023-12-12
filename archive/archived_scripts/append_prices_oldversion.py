import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm

stock_prices = pd.read_csv('stock_prices.csv')

result_df = pd.DataFrame()

for df in pd.read_csv('panel_removestocks.csv', chunksize=10000):

    for _, row in tqdm(df.iterrows()):
        start_date = pd.to_datetime(df.at[_, 'periodOfReport']).strftime('%Y-%m-%d')
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(years=1))
        symbol = df.at[_, 'issuerTradingSymbol']

        if symbol not in stock_prices['issuerTradingSymbol'].values:
            #print(f"Symbol '{symbol}' not found")
            continue
        #print(f"Now processing: '{symbol}'")
        mask = (stock_prices['issuerTradingSymbol'] == symbol)

        if start_date not in stock_prices.loc[mask, 'date'].values:
            #print(f"Symbol '{symbol}' has no start date price")
            df.at[_, 'init_prices'] = None
            continue
        df.at[_, 'init_prices'] = stock_prices[(stock_prices['date'] == start_date) & (stock_prices['issuerTradingSymbol'] == symbol)]['close'].values[0]

        max_days_to_search = 30

        while max_days_to_search > 0:
            end_date_str = end_date.strftime('%Y-%m-%d')
            if end_date_str in stock_prices.loc[mask, 'date'].values:
                df.at[_, 'end_dates'] = pd.to_datetime(end_date_str)
                df.at[_, 'end_prices'] = stock_prices[(stock_prices['date'] == end_date_str) & (stock_prices['issuerTradingSymbol'] == symbol)]['close'].values[0]
                break
            else:
                end_date += pd.DateOffset(days=1)
                max_days_to_search -= 1
        if max_days_to_search == 0:
            df.at[_, 'end_dates'] = None
            df.at[_, 'end_prices'] = None
            #print(f"Symbol '{symbol}' has no valid end price data")
    
    result_df = pd.concat([result_df, df])

result_df.to_csv('panel_cleaned_adddates.csv', index=False)