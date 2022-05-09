import numpy as np
import matplotlib.pyplot as plt

from utilities import load_data
from model import create_model, train


# test data 
name = 'eric'
# name = None

data, labels, data_length, labels_length = load_data(name)

model = create_model(data[0].shape[1], 10)
train(model, data, labels, data_length, labels_length, None)
