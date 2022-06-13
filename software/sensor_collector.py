import datetime
import os
import socket

import numpy as np
import pandas as pd

fileDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = os.path.dirname(fileDirectory)

DATA_RECORD_NAME = os.path.join(parentDirectory, 'data/numbers_data_record.csv')
DATA_DIR = os.path.join(parentDirectory, 'data/')   

def collector():
    """
    It opens a socket to the watch, waits 10 seconds, and then closes the socket
    """
    watch_data_exists = os.path.exists(DATA_DIR + 'raw_sensor_data.csv')
    if watch_data_exists:
        os.remove(DATA_DIR + 'raw_sensor_data.csv')

    start_time = datetime.datetime.now()
    #end time is 10 sec after the current time
    end_time = start_time + datetime.timedelta(seconds=10)

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('192.168.68.124', 59554)
    sock.connect(server_address)
    try:
        print('Opening socket')
        while end_time > datetime.datetime.now():
            data = sock.recv(4096)
            with open(DATA_DIR + 'raw_sensor_data.csv', 'ab') as f:
                f.write(data)
    finally:
        print('Closing socket')
        sock.close()
    cleaner()

def cleaner():
    """
    It reads in the raw data, drops the columns that are not needed, adds a new column that is the sum
    of the other columns, drops the other columns, and then appends the new data to the cleaned data
    file.
    """
    df = pd.read_csv(DATA_DIR + 'raw_sensor_data.csv')
    
    df = df.drop(columns=['loggingTime', 'accelerometerTimestamp_sinceReboot', 'gyroTimestamp_sinceReboot'])
    sum = df['accelerometerAccelerationX'] + df['accelerometerAccelerationY'] + df['accelerometerAccelerationZ'] + df['gyroRotationX'] + df['gyroRotationY'] + df['gyroRotationZ']
    df['imuSum'] = sum
    df = df.drop(columns=['accelerometerAccelerationX','accelerometerAccelerationY','accelerometerAccelerationZ','gyroRotationX','gyroRotationY','gyroRotationZ'])
    
    sensor_arr = df[['imuSum']].to_numpy().astype('float32')
    np.save(uniquify(DATA_DIR+'collected_data/collected_data.npy'), sensor_arr)
    
    print('\nRaw Data Cleaned & Appended Successfully.')

def uniquify(path):
    """
    It takes a path and returns a path that doesn't exist
    
    :param path: The path to the file you want to uniquify
    :return: The path to the file with a unique name.
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + "_" + extension
        counter += 1

    return path
