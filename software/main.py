from utilities import load_data
from model import create_model, train, test, save
from split import split
# from convert import convert_model_coreml

# test data 
# name = 'eric'
name = None

data, labels, data_length, labels_length = load_data(name)
data_train, labels_train, data_train_length, labels_train_length, data_test, labels_test, data_test_length, labels_test_length,  data_val, labels_val, data_val_length, labels_val_length = split(data, labels)

model = create_model(data[0].shape[1], 10)
train(model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length, None)

test(model, data_test, labels_test, data_test_length, labels_test_length)

save(model)

# Convert trained model to CoreML model for sensor log
# convert_model_coreml(SAVEDMODEL_DIR)
