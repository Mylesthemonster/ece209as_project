import matplotlib.pyplot as plt

from plot import plotter
from sensor_collector import collector
from utilities import load_data, chunk_data
from model import create_model, train, test, test_and_update, save_model, load_model
from tests import evaluate_gru_layer_sizes, run_model

# test data 
name = 'eric'
# name = None

# collector()

# train_data, val_data, test_data, chunked_train_data, chunked_val_data, chunked_test_data = load_data(name)

# data_train, labels_train, data_train_length, labels_train_length = train_data
# data_val, labels_val, data_val_length, labels_val_length = val_data
# data_test, labels_test, data_test_length, labels_test_length = test_data
# data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length = chunked_train_data
# data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length = chunked_val_data
# data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length = chunked_test_data

# train_model, test_model = create_model(data_train[0].shape[1], 10)
# # train_model_chunked, test_model_chunked = create_model(data_train_chunked[0].shape[1], 10)
# train(train_model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length)
# train(train_model_chunked, data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length, data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length)
# save_model(train_model, 'model.h5')
# save_model(train_model_chunked, 'model_chunked.h5')
# load_model(test_model, 'model.h5')
# load_model(test_model_chunked, 'model_chunked.h5')
# test_and_update(test_model, data_test, labels_test, data_test_length, labels_test_length)
# test(test_model, data_test, labels_test, data_test_length, labels_test_length)
# test(test_model_chunked, data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length)

# evaluate_gru_layer_sizes('accuracies.pkl')
run_model()

# Convert trained model to CoreML model for sensor log
# convert_model_coreml(SAVEDMODEL_DIR)

# plt.show()
