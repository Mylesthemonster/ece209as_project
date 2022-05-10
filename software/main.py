import numpy as np
import matplotlib.pyplot as plt

from utilities import load_data
from model import create_model, train
from split import split


# test data 
name = 'eric'
# name = None

data, labels, data_length, labels_length = load_data(name)
data_train, labels_train, data_test, labels_test, data_val, labels_val = split(data, labels)

model = create_model(data[0].shape[1], 10)
train(model, data, labels, data_length, labels_length, None)
