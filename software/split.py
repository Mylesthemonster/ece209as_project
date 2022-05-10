from re import X
from tkinter import Y
from sklearn.model_selection import train_test_split

# Split data into 80:10:10   TEST:TRAIN:VALIDATION

TRAIN_SIZE = 0.8 # Test set 80%
TEST_SIZE = 0.5  # Train 10% and Validation 10%

def split(data, labels):
    
    data_train, data_rem, labels_train, labels_rem = train_test_split(data, labels, train_size=TRAIN_SIZE)
    data_val, data_test, labels_val, labels_test = train_test_split(data_rem, labels_rem, test_size=TEST_SIZE)

    print(f'Train Data: %s' % (data_train.shape,))
    print(f'Train Labels: %s' % (labels_train.shape,))
    
    print(f'Test Data: %s' % (data_test.shape,))
    print(f'Test Labels: %s' % (labels_test.shape,))
    
    print(f'Validation Data: %s' % (data_val.shape,))
    print(f'Validation Labels: %s' % (labels_val.shape,))
    
    return data_train, labels_train, data_test, labels_test, data_val, labels_val

