import pandas as pd
from iexfinance.stocks import get_historical_data
from datetime import datetime, timedelta
import time

def get_data(symbols, dates):


    data = pd.DataFrame({'symbols': symbols, 'start_dates': dates, 'end_dates': None, 'init_prices': None, 'end_prices': None})

    earliest_date = min(dates)

    data['end_dates'] = pd.to_datetime(data['start_dates']) + pd.DateOffset(years=1)

    # for date in data['start_dates']:
    #     data['end_dates'].append((date + timedelta(days=365)).strftime('%Y-%m-%d'))


    try:
        #end_date = date + 1 year
        #get single stock
        #df = get_historical_data(symbol, date, end_date, output_format='pandas', close_only=True, token='pk_39af642288a040c99f76fd6bc833177f')

        json = get_historical_data(symbols=data['symbols'].values.tolist(), start=earliest_date, output_format='json', close_only=True, token='pk_39af642288a040c99f76fd6bc833177f')
        df = pd.DataFrame(json)

        for _, row in data.iterrows():
            if row['symbols'] not in df.columns:
                print(f"Symbol '{row['symbols']}' not found")
                continue
            print(f"Now processing: '{row['symbols']}'")
            stock_data = df[row['symbols']]
            stock_data = stock_data.explode('chart')
            stock_data = pd.json_normalize(stock_data)
            if 'date' not in stock_data:
                print(f"Symbol '{row['symbols']}' has no date")
                continue
            stock_data.set_index('date', inplace=True)
            start = pd.to_datetime(row['start_dates']).strftime('%Y-%m-%d')
            end = pd.to_datetime(row['end_dates'])
            if start in stock_data.index:
                data.at[_, 'init_prices'] = stock_data.loc[start, 'close']
            else:
                data.at[_, 'init_prices'] = None
            max_days_to_search = 30
            while max_days_to_search > 0:
                end_date_str = end.strftime('%Y-%m-%d')
                if end_date_str in stock_data.index:
                    data.at[_, 'end_dates'] = pd.to_datetime(end_date_str)
                    data.at[_, 'end_prices'] = stock_data.loc[end_date_str, 'close']
                    break
                else:
                    end += pd.DateOffset(days=1)
                    max_days_to_search -= 1
            if max_days_to_search == 0:
                data.at[_, 'end_dates'] = None
                data.at[_, 'end_prices'] = None
                print(f"Symbol '{row['symbols']}' has no valid end price data")



        # a_flat = pd.DataFrame(json)
        # tsla_data = a_flat['TSLA']
        # tsla = tsla_data.explode('chart')
        # tsla = pd.json_normalize(tsla)
        # tsla.set_index('date', inplace=True)
        # close_price = tsla.loc[date, 'close']

        return data

    except Exception as e:
        print(f"Error fetching data for {symbols}: {e}")
        return None

combined_data = pd.DataFrame()


#number of unique stocks is 18078
#could reduce this to only be top like 500 or something most recurring stocks...
# >>> panel['issuerTradingSymbol'].value_counts().head(50).sum()
# 446805
# >>> panel['issuerTradingSymbol'].value_counts().head(100).sum()
# 675761
# >>> panel['issuerTradingSymbol'].value_counts().head(500).sum()
# 1825977
# >>> panel['issuerTradingSymbol'].value_counts().head(1000).sum()
# 2744611
# >>> panel['issuerTradingSymbol'].value_counts().head(2000).sum()
# 3932500
# >>> panel['issuerTradingSymbol'].value_counts().head(3000).sum()
# 4668326
# >>> panel['issuerTradingSymbol'].value_counts().head(5000).sum()
# 5463473
# >>> panel['issuerTradingSymbol'].value_counts().head(7000).sum()
# 5820586
# >>> panel['issuerTradingSymbol'].value_counts().head(9000).sum()
# 5976274

for df in pd.read_csv('test_panelv2.csv', chunksize=100):
    symbols = []
    dates = []
    for symbol in df['issuerTradingSymbol']:
        symbols.append(symbol)
    for date in df['periodOfReport']:
        dates.append(pd.to_datetime(date).strftime('%Y-%m-%d'))

    try:
        data = get_data(symbols, dates)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
        print(combined_data.shape)
        merged_df = pd.concat([df, data[['end_dates', 'init_prices', 'end_prices']],], axis=1)
    except Exception as e:
        print(f"Error fetching data for Symbol: {symbols}, Date: {dates}: {e}")
    time.sleep(10)

pd.set_option('display.max_columns', None)
print(merged_df)
#merged_df.to_csv('test_panelv3.csv', index=False)


#load chunks