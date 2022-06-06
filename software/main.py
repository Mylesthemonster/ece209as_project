from utilities import load_data
from model import create_model, train, test, save
from split import split
import tensorflow.keras as keras
import os.path

# from convert import convert_model_coreml

# test data 
# name = 'eric'
name = None

model_exists = os.path.exists('saved_model.h5')

data, labels, data_length, labels_length = load_data(name)
data_train, labels_train, data_train_length, labels_train_length, data_test, labels_test, data_test_length, labels_test_length,  data_val, labels_val, data_val_length, labels_val_length = split(data, labels)

if model_exists:
    model = keras.models.load_model('saved_model.h5')
    test(model, data_test, labels_test, data_test_length, labels_test_length)
else: 
    model = create_model(data[0].shape[1], 10)
    train(model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length, None)
    save(model)

# Convert trained model to CoreML model for sensor log
# convert_model_coreml(SAVEDMODEL_DIR)
