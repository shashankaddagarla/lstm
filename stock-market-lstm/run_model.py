# takes care of the actual training and running of the model

import load_data, model, plot_data

import time
import threading
import gc

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

num_epochs = 5
num_neurons_l1 = 128
num_neurons_l2 = 128
train_test_split = 0.9

batch_size = 5000
num_examples = 8295
num_inputs = 10
num_features = 5
np.random.seed(420)

def fit_model_threaded(model, data_gen, steps_per_epoch):
	model.fit_generator(data_gen, steps_per_epoch, num_epochs)
	return

gc.collect()
pd.options.mode.chained_assignment = None
'''
df = load_data.create_dataframe_for(how='kaggle', ticker='HPQ')
df = load_data.create_dataframe_for(how="kaggle", ticker='AAPL')
df = df.sort_values('Date')

cleaned_data = load_data.clean_data(df, 10, 3)
load_data.write_clean_data(cleaned_data, filename='kaggle-aapl-clean.npz')
'''
gen_train = load_data.gen_clean_data(filename='kaggle-aapl-clean.npz', set='train', batch_size = batch_size)
gen_test = load_data.gen_clean_data(filename='kaggle-aapl-clean.npz', set='test', batch_size = batch_size)

training_size = int(num_examples * train_test_split)
test_size = num_examples - training_size

steps_per_epoch = int(training_size / batch_size)
print('>>> Clean data has', num_examples, 'data rows. Training on', training_size, 'training examples with', steps_per_epoch, 'steps-per-epoch')

model = model.build_model([num_inputs, num_features, num_neurons_l1, num_neurons_l2, 1])
model.fit_generator(gen_train, steps_per_epoch, num_epochs)

steps_test = int(test_size / batch_size)
print('>>> Testing model on', test_size, 'data rows with', steps_test, 'steps')
predictions = model.predict_generator(gen_test, steps=steps_test)

plt.figure(figsize = (18,12))
plt.plot(predictions)
plt.show()


