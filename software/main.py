import os

import matplotlib.pyplot as plt
import numpy as np

from model import (create_model, load_model, save_model, test, test_and_update,
                   train)
from plot import plotter
from sensor_collector import collector
from tests import evaluate_gru_layer_sizes, run_model
from utilities import chunk_data, load_data

fileDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = os.path.dirname(fileDirectory)

DATA_DIR = os.path.join(parentDirectory, 'data/')  

# test data 
name = 'eric' # subset of data
# name = None # all the data

# COLLECT DATA
# collector()

# LOAD DATASET
train_data, val_data, test_data, chunked_train_data, chunked_val_data, chunked_test_data = load_data(name)

data_train, labels_train, data_train_length, labels_train_length = train_data
data_val, labels_val, data_val_length, labels_val_length = val_data
data_test, labels_test, data_test_length, labels_test_length = test_data
data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length = chunked_train_data
data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length = chunked_val_data
data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length = chunked_test_data

# CREATE AND TRAIN MODEL
train_model, test_model = create_model(data_train[0].shape[1], 10)
train(train_model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length)
save_model(train_model, DATA_DIR + 'model.h5')

# LOAD WEIGHTS INTO TEST MODEL
load_model(test_model, DATA_DIR + 'model.h5')
# test_and_update(test_model, data_test, labels_test, data_test_length, labels_test_length)
test(test_model, data_test, labels_test, data_test_length, labels_test_length)

# ALL of this can be run using run_model

# Convert trained model to CoreML model for sensor log
# convert_model_coreml(SAVEDMODEL_DIR)
