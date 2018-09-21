# 10-16-18
#
# currently a two-layer lstm model with dropouts (default .25 dropout) after each layer and a dense layer at the end to produce output

import os
import time

import h5py
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers

activation_function = 'tanh'
loss = 'mae'
optimizer = optimizers.Adam(clipnorm = 1)
dropout = .25

def build_model(layers, activ_func=activation_function, dropout=dropout, optimizer=optimizer):
	model = Sequential()

	model.add(LSTM(input_shape = (layers[0], layers[1]), return_sequences = True, units= layers[2])) # first layer so required input_shape
	model.add(Dropout(dropout))

	model.add(LSTM(layers[3], return_sequences = False, activation = activ_func))
	model.add(Dropout(dropout))

	model.add(Dense(units=layers[4]))
	model.add(Activation(activ_func))

	start = time.time()
	model.compile(loss = loss, optimizer = optimizer)
	print('>>> Model compiled! Took {} seconds.'.format(time.time() - start))
	return model

def save_model(model, name='my_model'):
	model.save(filename+'.h5')
	del model

def load_model(name):
	if(os.path.isfile(name+'.h5')):
		return load_model(name+'.h5')
	else:
		print('>>> The specified model cannot be found.')
		return None