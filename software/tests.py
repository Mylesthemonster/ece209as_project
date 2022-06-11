from model import create_model, train, test
from utilities import load_data
import pickle
import matplotlib.pyplot as plt
import os

def evaluate_gru_layer_sizes(gru_sizes=[32, 128], gru_step=16, name='eric'):
    accuracies = {}

    if not os.path.exists('gru_accuracies.pkl'):
        train_data, val_data, test_data, chunked_train_data, chunked_val_data, chunked_test_data = load_data(name)

        data_train, labels_train, data_train_length, labels_train_length = train_data
        data_val, labels_val, data_val_length, labels_val_length = val_data
        data_test, labels_test, data_test_length, labels_test_length = test_data
        for i in range(gru_sizes[0], gru_sizes[1], gru_step):
            train_model, test_model = create_model(data_train[0].shape[1], 10)
            train(train_model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length)
            accuracies[i] = test(test_model, data_test, labels_test, data_test_length, labels_test_length)
        
        with open('gru_accuracies.pkl','wb') as f:
            pickle.dump(accuracies, f)
    else:
        with open('gru_accuracies.pkl','rb') as f:
            accuracies = pickle.load(f)

    plt.plot(accuracies.keys(), accuracies.values(), '-o')
    plt.title('GRU layer sizes vs Model accuracy')
    plt.xlabel('GRU layer units')
    plt.ylabel('Accuracy')
    plt.show()

def run_model(name='eric', plot=False):
    train_data, val_data, test_data, chunked_train_data, chunked_val_data, chunked_test_data = load_data(name)

    data_train, labels_train, data_train_length, labels_train_length = train_data
    data_val, labels_val, data_val_length, labels_val_length = val_data
    data_test, labels_test, data_test_length, labels_test_length = test_data
    
    train_model, test_model = create_model(data_train[0].shape[1], 10)
    train(train_model, data_train, labels_train, data_train_length, labels_train_length, data_val, labels_val, data_val_length, labels_val_length, plot)
    acc = test(test_model, data_test, labels_test, data_test_length, labels_test_length)

    print(acc)