# takes care of the actual training and running of the model

import load_data, model, plot_data

import time
import threading

import numpy as np
import pandas as pd
import h5py

gc.collect()