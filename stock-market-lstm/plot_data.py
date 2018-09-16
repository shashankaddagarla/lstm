import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def plot_raw_data(df):
	plt.figure(figsize = (18,9))
	plt.plot(range(df.shape[0]),(df['Low']+df['High']+df['Open']+df['Close'])/4.0)
	stepsize = int(len(df)/24)
	plt.xticks(range(0,df.shape[0],stepsize),df['Date'].loc[::stepsize],rotation=45)
	plt.xlabel('Date',fontsize=14)
	plt.ylabel('OHLC Average Price',fontsize=14)
	plt.show()