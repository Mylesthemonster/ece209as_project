import os
import pickle

import matplotlib.pyplot as plt

from model import create_model, test, train
from utilities import load_data

fileDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = os.path.dirname(fileDirectory)

DATA_DIR = os.path.join(parentDirectory, 'data/')  

def evaluate_gru_layer_sizes(gru_sizes=[32, 128], gru_step=16, name='eric'):
    """
    It takes in a range of GRU layer sizes, and for each size, it trains a model and tests it on the
    test data
    
    :param gru_sizes: The range of GRU layer sizes to test
    :param gru_step: The step size for the GRU layer sizes, defaults to 16 (optional)
    :param name: the name of the person whose data you want to use, defaults to eric (optional)
    """
    accuracies = {}

    if not os.path.exists(DATA_DIR + 'gru_accuracies.pkl'):
        train_data, val_data, test_data, chunked_train_data, chunked_val_data, chunked_test_data = load_data(name)

        data_train, labels_train, data_train_length, labels_train_length = train_data
        data_val, labels_val, data_val_length, labels_val_length = val_data
        data_test, labels_test, data_test_length, labels_test_length = test_data
        for i in range(gru_sizes[0], gru_sizes[1], gru_step):
            train_model, test_model = create_model(data_train[0].shape[1], 10)
            train(train_model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length)
            accuracies[i] = test(test_model, data_test, labels_test, data_test_length, labels_test_length)
        
        with open(DATA_DIR + 'gru_accuracies.pkl','wb') as f:
            pickle.dump(accuracies, f)
    else:
        with open(DATA_DIR + 'gru_accuracies.pkl','rb') as f:
            accuracies = pickle.load(f)

    plt.plot(accuracies.keys(), accuracies.values(), '-o')
    plt.title('GRU layer sizes vs Model accuracy')
    plt.xlabel('GRU layer units')
    plt.ylabel('Accuracy')
    plt.show()

def run_model(name='eric', plot=False):
    """
    It loads the data, creates the model, trains the model, and tests the model
    
    :param name: the name of the person you want to train on, defaults to eric (optional)
    :param plot: whether to plot the training and validation loss, defaults to False (optional)
    """
    train_data, val_data, test_data, chunked_train_data, chunked_val_data, chunked_test_data = load_data(name)

    data_train, labels_train, data_train_length, labels_train_length = train_data
    data_val, labels_val, data_val_length, labels_val_length = val_data
    data_test, labels_test, data_test_length, labels_test_length = test_data
    
    train_model, test_model = create_model(data_train[0].shape[1], 10)
    train(train_model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length, plot)
    acc = test(test_model, data_test, labels_test, data_test_length, labels_test_length)

    print(acc)
