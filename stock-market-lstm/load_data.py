import pandas as pd
import urllib.request, json
import datetime as dt
import os

alphavantage_api_key = 'LEUKYYXO15GYK7Y3'

def load_data(how='kaggle', ticker='AAL'):
	if how == 'alphavantage':
		url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker, alphavantage_api_key)
		file_name = "data/stock_market_data-%s.csv"%ticker
		if not os.path.exists(file_name):
			with urllib.request.urlopen(url_string) as url:
				data = json.loads(url.read().decode())
				data = data['Time Series (Daily)']
				df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
				for key, val in data.items():
					date = dt.datetime.strptime(key, '%Y-%m-%d')
					data_row = [date.date(), float(val['1. open']), float(val['2. high']), float(val['3. low']), float(val['4. close']), float(val['5. volume'])]
					df.loc[-1,:] = data_row
					df.index += 1
			print('Data saved to %s'%file_name)
			df.to_csv(file_name)
		else:
			print('File already exists. Loading data from %s'%file_name)
			df = pd.read_csv(file_name)
	elif how == 'kaggle':
		print('Loading data from Kaggle repository')

load_data(how='alphavantage')
