import numpy as np
import pandas as pd
import os
import time
from datetime import timedelta

start_time = time.time()

fileDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = os.path.dirname(fileDirectory)

DATA_RECORD_NAME = os.path.join(parentDirectory, 'data/numbers_data_record.csv')
DATA_DIR = os.path.join(parentDirectory, 'data/numbers/')   

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

    data_length = np.array([len(e) for e in data]).astype(np.int64)
    labels_length = np.array([len(l) for l in labels]).astype(np.int64)
    data = pad_data(np.array(data, dtype=object), data_length)
    labels = pad_labels(np.array(labels, dtype=object), labels_length)
    
    print(f'\nDone Loading Data it took: {timedelta(seconds = (time.time() - start_time))}')

    return data, labels, data_length, labels_length

def pad_data(data, data_length):
    """
    It takes a list of numpy arrays and pads them with zeros to make them all the same length
    
    :param data: the data to be padded
    :param data_length: the length of each data point
    :return: The padded data is being returned.
    """
    max_data_length = data_length.max()
    padded_data = np.zeros((data.shape[0], max_data_length, data[0].shape[1]))
    for i, d in enumerate(data):
        padded_data[i, :d.shape[0], :] = d

    return padded_data

def pad_labels(labels, labels_length):
    """
    It takes a list of lists of labels and a list of the lengths of each list of labels, and returns a
    2D array of labels padded with zeros
    
    :param labels: the labels for each of the training examples
    :param labels_length: the length of each label
    :return: The padded_labels are being returned.
    """
    max_labels_length = labels_length.max()
    padded_labels = np.zeros((labels.shape[0], max_labels_length))
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l

    return padded_labels
