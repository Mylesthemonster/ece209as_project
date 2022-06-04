import numpy as np
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta

# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(linewidth=100)
start_time = time.time()

# Split data into 80:10:10   TEST:TRAIN:VALIDATION

TRAIN_SIZE = 0.8 # Test set 80%
TEST_SIZE = 0.5  # Train 10% and Validation 10%

def split(data, labels):
    """
    The function takes in the data and labels and splits them into training, validation, and test sets.
    
    :param data: the data we want to split
    :param labels: The labels of the data
    """
    print('\nSpliting Data...')
    
    data_train, data_rem, labels_train, labels_rem = train_test_split(data, labels, train_size=TRAIN_SIZE)
    data_val, data_test, labels_val, labels_test = train_test_split(data_rem, labels_rem, test_size=TEST_SIZE)

    data_train_length = np.array([len(a) for a in data_train]).astype(np.int64)
    labels_train_length = np.array([len(b) for b in labels_train]).astype(np.int64)
    
    data_test_length = np.array([len(f) for f in data_test]).astype(np.int64)
    labels_test_length = np.array([len(g) for g in labels_test]).astype(np.int64)
    
    data_val_length = np.array([len(c) for c in data_val]).astype(np.int64)
    labels_val_length = np.array([len(e) for e in labels_val]).astype(np.int64)
    
    print(f'\nData Split it took: {timedelta(seconds = (time.time() - start_time))}')
    
    # print(f'\nTrain Data shape: %s' % (data_train.shape,))
    # print(f'Train Labels shape: %s' % (labels_train.shape,))
    # print(f'Length Train Data size: %s' % (data_train.size,))
    # print(f'Length Train Labels size: %s' % (labels_train.size,))
    # print(f'data_train_length shape: %s' % (data_train_length.shape,))
    # print(f'data_train_length size: %s' % (data_train_length.size,))
    # print(f'labels_train_length shape: %s' % (labels_train_length.shape,))
    # print(f'labels_train_length size: %s' % (labels_train_length.size,))
   
    # print(f'\nTest Data shape: %s' % (data_test.shape,))
    # print(f'Test Labels shape: %s' % (labels_test.shape,))
    # print(f'Length Test Data size: %s' % (data_test.size,))
    # print(f'Length Test Labels size: %s' % (labels_test.size,))
    # print(f'data_test_length shape: %s' % (data_test_length.shape,))
    # print(f'data_test_length size: %s' % (data_test_length.size,))
    # print(f'labels_test_length shape: %s' % (labels_test_length.shape,))
    # print(f'labels_test_length size: %s' % (labels_test_length.size,))
    
    # print(f'\nValidation Data shape: %s' % (data_val.shape,))
    # print(f'Validation Labels shape: %s' % (labels_val.shape,))
    # print(f'Length Data size: %s' % (data_val.size,))
    # print(f'Length Labels size: %s' % (labels_val.size,))
    # print(f'data_val_length shape: %s' % (data_val_length.shape,))
    # print(f'data_val_length size: %s' % (data_val_length.size,))
    # print(f'labels_val_length shape: %s' % (labels_val_length.shape,))
    # print(f'labels_val_length size: %s' % (labels_val_length.size,))
    
    return data_train, labels_train, data_train_length, labels_train_length, data_test, labels_test, data_test_length, labels_test_length,  data_val, labels_val, data_val_length, labels_val_length

