import os
import time
from datetime import timedelta
from chunk import chunk_input_stream

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

    data_train, data_rem, labels_train, labels_rem = train_test_split(data, labels, train_size=TRAIN_SIZE)
    data_val, data_test, labels_val, labels_test = train_test_split(data_rem, labels_rem, test_size=TEST_SIZE)

    data_train_length = np.array([len(e) for e in data_train]).astype(np.int64)
    labels_train_length = np.array([len(l) for l in labels_train]).astype(np.int64)
    data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length = \
        chunk_data(data_train, labels_train, data_train_length, labels_train_length)
    data_train_chunked, labels_train_chunked = pad_data(data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length)
    data_train, labels_train = pad_data(data_train, labels_train, data_train_length, labels_train_length)

    data_val_length = np.array([len(e) for e in data_val]).astype(np.int64)
    labels_val_length = np.array([len(l) for l in labels_val]).astype(np.int64)
    data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length = \
        chunk_data(data_val, labels_val, data_val_length, labels_val_length)
    data_val_chunked, labels_val_chunked = pad_data(data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length)
    data_val, labels_val = pad_data(data_val, labels_val, data_val_length, labels_val_length)

    data_test_length = np.array([len(e) for e in data_test]).astype(np.int64)
    labels_test_length = np.array([len(l) for l in labels_test]).astype(np.int64)
    data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length = \
        chunk_data(data_test, labels_test, data_test_length, labels_test_length)
    data_test_chunked, labels_test_chunked = pad_data(data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length)
    data_test, labels_test = pad_data(data_test, labels_test, data_test_length, labels_test_length)
    
    print(f'\nData Loaded & Partitioned, it took: {timedelta(seconds = (time.time() - start_time))}')

    return (data_train, labels_train, data_train_length, labels_train_length), \
        (data_val, labels_val, data_val_length, labels_val_length), \
        (data_test, labels_test, data_test_length, labels_test_length), \
        (data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length), \
        (data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length), \
        (data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length)


def pad_data(data, labels, data_length, labels_length):
    max_data_length = data_length.max()
    padded_data = np.zeros((data.shape[0], max_data_length, data[0].shape[1]))
    for i, d in enumerate(data):
        padded_data[i, :d.shape[0], :] = d

    max_labels_length = labels_length.max()
    padded_labels = np.zeros((labels.shape[0], max_labels_length))
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l[:labels_length[i]]

    return padded_data, padded_labels

def chunk_data(input_data, labels, data_length, labels_length):
    """
    It takes the data and labels and divides the data into chunks of the length specified by the labels
    
    :param data_train: The training data
    :param labels_train_length: The length of each sequence in the training set
    :param data_val: The validation data
    :param labels_val_length: The length of each sequence in the validation set
    :param data_test: The test data
    :param labels_test_length: The length of each sequence in the test set
    """

    new_data = []
    new_labels = []
    new_data_lengths = []
    new_labels_lengths = []

    for i in range(len(input_data)):
        if labels_length[i] != 4:
            continue
        data = input_data[i]
        
        window_size = 300 / labels_length[i] * 2
        overlap_size = 0
        chunk_size = int(window_size + overlap_size)
        stride = int(window_size)

        n_chunks = 300 // stride

        new_input_seq = np.zeros((n_chunks, len(data) * n_chunks, 300))
        new_label_seq = np.zeros((n_chunks, 2))
        new_label_seq[0] = labels[i][:2]
        new_label_seq[1] = labels[i][2:]

        for seq_i in range(len(data)):
            seq = data[seq_i]
            chunked = [seq[i:i + chunk_size] for i in range(0, len(seq), stride)]
            for j in range(len(chunked)):
                chunk = chunked[j]
                # if chunk.shape[0] != chunk_size:
                    # chunk = np.pad(chunk, (0, chunk_size - chunk.shape[0]), 'constant', constant_values=(0, 0))
                chunk = np.pad(chunk, (0, 300 - chunk.shape[0]), 'constant', constant_values=(0, 0))
            
                new_input_seq[j, seq_i, :] = chunk
        
        new_data.extend(new_input_seq)
        new_labels.extend(new_label_seq)
        new_labels_lengths.extend([2] * n_chunks)
        new_data_lengths.extend([len(data) * n_chunks] * n_chunks)

    print(f'\nData Divided into Chunks, it took: {timedelta(seconds = (time.time() - start_time))}')
    
    return np.array(new_data), np.array(new_labels), np.array(new_data_lengths), np.array(new_labels_lengths)
