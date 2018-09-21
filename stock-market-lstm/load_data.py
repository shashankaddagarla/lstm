import urllib.request, json
import datetime as dt
import os

import plot_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
			print('>>> Data saved to %s'%file_name)
			df.to_csv(file_name, index=False)
		else:
			print('>>> File already exists. Loading data from %s'%file_name)
			df = pd.read_csv(file_name)
		return df
	elif how == 'kaggle':
		print('>>> Loading data from Kaggle repository!')
		df = pd.read_csv(os.path.join('kaggle-data/Stocks', '%s.us.txt')%ticker.lower())
		return df
	else:
		print('>>> Cannot get data from specified source.')
		return None

# normalize according to formula n_i = p_i/p_0 - 1, p being raw price info
def normalize_window(df):
	return (df / df.iloc[0]) - 1

def clean_data(df, x_window_size, y_window_size, normalize=True):
	if 'OpenInt' in df.columns: df.drop(['OpenInt'], axis = 1, inplace = True)
	num_days = len(df)
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
		print(index)
		index += 1

	x_data_3d = np.array(x_data)
	y_data_3d = np.array(y_data)
	return (x_data_3d, y_data_3d)

def verify_data(data):
	for x, y in (data):
		print(np.where(x > 2))

def write_clean_data(data, test_train_split = 0.9, filename='clean-data.npz'):
	num_train_examples = int((data[0].shape[0]) * test_train_split)
	np.savez_compressed(
		file=os.path.join(os.curdir, 'processed-data', filename),
		x_train=data[0][:num_train_examples],
		x_test=data[0][num_train_examples:],
		y_train=data[1][:num_train_examples],
		y_test=data[1][num_train_examples:]
	)
	print('>>> Processed data written to', filename)

def gen_clean_data(filename, set='train', batch_size = 1000):
	index = 0
	if os.path.isfile(os.path.join(os.curdir, 'processed-data', filename)):
		data = np.load(os.path.join(os.curdir, 'processed-data', filename))
		print('>>> Loaded data from', filename)
	else:
		print('>>> Processed data with specified filename not found.')
		return None
	x_train = data['x_train']
	x_test = data['x_test']
	y_train = data['y_train']
	y_test = data['y_test']

	num_examples = x_train.shape[0] if set == 'train' else x_test.shape[0]

	while (True):
		index += batch_size
		if (set == 'train'):
			if (index > num_examples):
				index = 0
				yield (x_train[num_examples - (num_examples % batch_size):num_examples], y_train[num_examples - (num_examples % batch_size):num_examples])
			elif (index == num_examples):
				index = 0
				yield (x_train[num_examples - batch_size:num_examples], y_train[num_examples - batch_size:num_examples])
			else:
				yield (x_train[index - batch_size:index], y_train[index - batch_size:index])
		elif (set == 'test'):
			if (index >= num_examples):
				index = 0
				yield (x_test[num_examples - (num_examples % batch_size):num_examples], y_test[num_examples - (num_examples % batch_size):num_examples])
			else:
				yield (x_test[index - batch_size:index], y_test[index - batch_size:index])
		else:
			print('>>> Not a valid dataset.')
			return None

# df = create_dataframe_for(how='kaggle', ticker='HPQ')
# df = create_dataframe_for(how="kaggle", ticker='AAPL')
# df = df.sort_values('Date')

# cleaned_data = clean_data(df, 50, 20)
# write_clean_data(cleaned_data, filename='kaggle-aapl-clean.npz')
'''
data = np.load(os.path.join(os.curdir, 'processed-data', 'kaggle-aapl-clean.npz'))
whatwewant = zip(data['x_train'], data['y_train'])
verify_data(whatwewantsource)'''