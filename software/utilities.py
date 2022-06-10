import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

start_time = time.time()

fileDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = os.path.dirname(fileDirectory)

DATA_RECORD_NAME = os.path.join(parentDirectory, 'data/numbers_data_record.csv')
DATA_DIR = os.path.join(parentDirectory, 'data/numbers/')   

TRAIN_SIZE = 0.8 # Test set 80%
TEST_SIZE = 0.5  # Train 10% and Validation 10%

def load_data(name):
    """
    It loads the data from the csv file, and then loads the data from the numpy files
    
    :param name: The name of the dataset to load. If None, all datasets will be loaded
    :return: data, labels, data_length, labels_length
    """
    print('\nLoading Data....')
    data_record = pd.read_csv(DATA_RECORD_NAME, converters={'labels': pd.eval})
    data_record['data_path'] = data_record['data_path'].str.replace('\\', '/', regex=False) # update path syntax
    data_record['data_path'] = data_record['data_path'].str.replace('../train_data/numbers/', DATA_DIR, regex=False) # rename directory
    if name is not None: # filter dataset by name
        data_record = data_record[data_record['name'] == name]
    data = []
    labels = []
    for index, record in data_record.iterrows():
        raw_data = np.load(record['data_path']).astype(np.float32)
        raw_labels = np.array(record['labels']).astype(np.float32)
        data.append(raw_data)
        labels.append(raw_labels)

    data = np.array(data, dtype=object)
    labels = np.array(labels, dtype=object)

    # data_length = np.array([len(e) for e in data]).astype(np.int64)
    # labels_length = np.array([len(l) for l in labels]).astype(np.int64)
    # data = pad_data(np.array(data, dtype=object), data_length)
    # labels = pad_labels(np.array(labels, dtype=object), labels_length)

    data_train, data_rem, labels_train, labels_rem = train_test_split(data, labels, train_size=TRAIN_SIZE)
    data_val, data_test, labels_val, labels_test = train_test_split(data_rem, labels_rem, test_size=TEST_SIZE)

    data_train_length = np.array([len(e) for e in data_train]).astype(np.int64)
    labels_train_length = np.array([len(l) for l in labels_train]).astype(np.int64)
    data_train, labels_train = pad_data(data_train, labels_train, data_train_length, labels_train_length)

    data_val_length = np.array([len(e) for e in data_val]).astype(np.int64)
    labels_val_length = np.array([len(l) for l in labels_val]).astype(np.int64)
    data_val, labels_val = pad_data(data_val, labels_val, data_val_length, labels_val_length)

    data_test_length = np.array([len(e) for e in data_test]).astype(np.int64)
    labels_test_length = np.array([len(l) for l in labels_test]).astype(np.int64)
    data_test, labels_test = pad_data(data_test, labels_test, data_test_length, labels_test_length)
    
    print(f'\nData Loaded & Partitioned, it took: {timedelta(seconds = (time.time() - start_time))}')

    return (data_train, labels_train, data_train_length, labels_train_length), \
        (data_val, labels_val, data_val_length, labels_val_length), \
        (data_test, labels_test, data_test_length, labels_test_length)


def pad_data(data, labels, data_length, labels_length):
    max_data_length = data_length.max()
    padded_data = np.zeros((data.shape[0], max_data_length, data[0].shape[1]))
    for i, d in enumerate(data):
        padded_data[i, :d.shape[0], :] = d

    max_labels_length = labels_length.max()
    padded_labels = np.zeros((labels.shape[0], max_labels_length))
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l

    return padded_data, padded_labels
