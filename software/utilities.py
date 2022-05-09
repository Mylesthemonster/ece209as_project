import numpy as np
import pandas as pd

DATA_RECORD_NAME = 'data/numbers_data_record.csv'
DATA_DIR = 'data/numbers/'

def load_data(name):
    data_record = pd.read_csv(DATA_RECORD_NAME, converters={'labels': pd.eval})
    data_record['data_path'] = data_record['data_path'].str.replace('\\', '/') # update path syntax
    data_record['data_path'] = data_record['data_path'].str.replace('../train_data/numbers/', DATA_DIR) # rename directory
    records = data_record[data_record['name'] == name]
    data = []
    labels = []
    for index, record in records.iterrows():
        raw_data = np.load(record['data_path']).astype(np.float32)
        raw_labels = np.array(record['labels']).astype(np.float32)
        # if len(raw_labels) == 2:
        #     data.append(raw_data)
        #     labels.append(raw_labels)
        data.append(raw_data)
        labels.append(raw_labels)

    data_length = np.array([len(e) for e in data]).astype(np.int64)
    labels_length = np.array([len(l) for l in labels]).astype(np.int64)
    data = pad_data(np.array(data), data_length)
    labels = pad_labels(np.array(labels), labels_length)

    return data, labels, data_length, labels_length


def pad_data(data, data_length):
    max_data_length = data_length.max()
    padded_data = np.zeros((data.shape[0], max_data_length, data[0].shape[1]))
    for i, d in enumerate(data):
        padded_data[i, :d.shape[0], :] = d

    return padded_data

def pad_labels(labels, labels_length):
    max_labels_length = labels_length.max()
    print(labels.shape)
    padded_labels = np.zeros((labels.shape[0], max_labels_length))
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l

    return padded_labels
