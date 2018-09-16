import urllib.request, json
import datetime as dt
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

alphavantage_api_key = 'LEUKYYXO15GYK7Y3'

def create_dataframe_for(how='kaggle', ticker='AAL'):
	if how == 'alphavantage':
		url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker, alphavantage_api_key)
		file_name = "aa-data/stock_market_data-%s.csv"%ticker
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
		return df
	elif how == 'kaggle':
		print('Loading data from Kaggle repository!')
		df = pd.read_csv(os.path.join('kaggle-data/Stocks', '%s.us.txt')%ticker.lower())
		return df
	else:
		print('Cannot get data from specified source.')
		return None

# normalize according to formula n_i = p_i/p_0 - 1, p being raw price info
def normalize_window(df):
	return (df / df.iloc[0]) - 1

def gen_clean_data(df, x_window_size, y_window_size, batch_size=1000, normalize=True):

	df.drop(['OpenInt'], axis = 1, inplace = True)
	num_days = x_window_size + y_window_size + 1000
	ema_smoothing_constant = 2.0 / (y_window_size + 1)
	x_data = list()
	y_data = list()
	index = 0

	total_window_size = x_window_size + y_window_size

	while (index + total_window_size <= num_days):
		current_window = df[index:(index + total_window_size)]
		current_window.drop(['Date'], axis = 1, inplace = True)
		if (normalize):
			current_window = normalize_window(current_window)
		x_window = current_window[:x_window_size]
		y_window = current_window[x_window_size:]

		x_data.append(x_window.values)
		if y_data:
			y_data_ave = y_window['Close'].iloc[-1] * ema_smoothing_constant + y_data[-1] * (1 - ema_smoothing_constant)
		else:
			y_data_ave = y_window['Close'].mean()
		y_data.append(y_data_ave)

		index += 1
		if (index % batch_size == 0):
			x_data_3d = np.array(x_data)
			y_data_3d = np.array(y_data)
			x_data = []
			y_data = []
			return (x_data_3d, y_data_3d)

def split_data(df, training_test_split = 0.9):
	return df[:int(training_test_split * len(df))], df[int(training_test_split * len(df)):]


df = create_dataframe_for(how='kaggle', ticker='HPQ')
df = df.sort_values('Date')
print(gen_clean_data(df, 50, 20))



''' plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Low']+df['High']+df['Open']+df['Close'])/4.0)
#plt.plot(range(ohlc_ave.shape[0]), ohlc_ave)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show() '''