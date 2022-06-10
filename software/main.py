import matplotlib.pyplot as plt

from model import create_model, save, test, train
from plot import plotter
from sensor_collector import collector
from utilities import load_data

# test data 
name = 'eric'
# name = None

collector()

train_data, val_data, test_data = load_data(name)

data_train, labels_train, data_train_length, labels_train_length = train_data
data_val, labels_val, data_val_length, labels_val_length = val_data
data_test, labels_test, data_test_length, labels_test_length = test_data

plotter(data_train, data_val, data_test)

# model = create_model(data_train[0].shape[1], 10)
# train(model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length, None)

# # save(model)

# test(model, data_test, labels_test, data_test_length, labels_test_length)


# Convert trained model to CoreML model for sensor log
# convert_model_coreml(SAVEDMODEL_DIR)

plt.show()
