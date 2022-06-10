import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np

start_time = time.time()

def plotter(data_train, data_val, data_test):
    """
    It takes in three lists of data, and plots them in a 2x2 grid
    
    :param data_train: The training data
    :param data_val: The validation data
    :param data_test: The test data
    """ 
    print('\nPlotting Data....')
    
    X = np.arange(0,data_train.shape[2],1)
   
    fig, ax = plt.subplots(3)
    
    x = 0
    y = 0
    # For Train data
    while y in range(data_train.shape[1] - 1):
        while x in range(data_train.shape[2] - 1):
            ax[0].plot(X,data_train[x][y])
            ax[0].set_title('Train data')
            x += 1
        y += 1
    
    x = 0
    y = 0
    # For Validation data
    while y in range(data_val.shape[1] - 1):
        while x in range(data_val.shape[2] - 1):
            ax[1].plot(X,data_val[x][y])
            ax[1].set_title('Validation data')
            x += 1
        y += 1
        
    x = 0
    y = 0
    # For Test data
    while y in range(data_test.shape[1] - 1):
        while x in range(data_test.shape[2] - 1):
            ax[2].plot(X,data_test[x][y])
            ax[2].set_title('Test data')
            x += 1
        y += 1
        
    plt.subplots_adjust(hspace=0.4)
    plt.draw()
    
    print(f'\nPlots Generated, it took: {timedelta(seconds = (time.time() - start_time))}')
