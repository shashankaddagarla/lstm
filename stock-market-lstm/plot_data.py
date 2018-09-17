import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def plot_raw_data(df):
	plt.figure(figsize = (18,12))
	plt.plot(range(df.shape[0]),(df['Low']+df['High']+df['Open']+df['Close'])/4.0)
	stepsize = int(len(df)/24)
	plt.xticks(range(0,df.shape[0],stepsize),df['Date'].loc[::stepsize],rotation=45)
	plt.xlabel('Date',fontsize=14)
	plt.ylabel('OHLC Average Price',fontsize=14)
	plt.show()

def plot_singlept_predictions(training_hist, train_data, train_predictions, test_data, test_predictions):
	plt.figure(figsize=(18, 12))
	plt.subplot(311)
	plt.plot(history.epoch, history.history['loss'])
	plt.plot(history.epoch, history.history['val_loss'])
	plt.xlabel('# of epochs')
	plt.ylabel('Loss')
	plt.title('Model Loss')
	plt.legend(['Training', 'Test'])

	plt.subplot(312)
	plt.plot(train_data)
	plt.plot(train_predictions)
	plt.xlabel('Date')
	plt.ylabel('Price History')
	plt.title('Single Point Prediction on Training Set')
	plt.legend(['Actual', 'Predicted'])

	plt.subplot(313)
	plt.plot(test_data)
	plt.plot(test_predictions)
	plt.xlabel('Dates')
	plt.xlabel('Date')
	plt.ylabel('Price History')
	plt.title('Single Point Prediction on Test Set')
	plt.legend(['Actual', 'Predicted'])

	plt.show()

def plot_sequence_predictions(training_hist, train_data, train_predictions, test_data, test_predictions, prediction_len):
	fig = plt.figure(figsize=(18, 12))
	plt.subplot(311)
	plt.plot(history.epoch, history.history['loss'])
	plt.plot(history.epoch, history.history['val_loss'])
	plt.xlabel('# of epochs')
	plt.ylabel('Loss')
	plt.title('Model Loss')
	plt.legend(['Training', 'Test'])

	plt.subplot(312)
	plt.plot(train_data)
	for i, data in enumerate(train_predictions):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data)
	plt.xlabel('Date')
	plt.ylabel('Price History')
	plt.title('Single Point Prediction on Training Set')
	plt.legend(['Actual', 'Predicted'])

	plt.subplot(313)
	plt.plot(test_data)
	for i, data in enumerate(test_predictions):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data)
	plt.xlabel('Dates')
	plt.xlabel('Date')
	plt.ylabel('Price History')
	plt.title('Single Point Prediction on Test Set')
	plt.legend(['Actual', 'Predicted'])

	plt.show()

