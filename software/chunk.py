import os
from itertools import islice

import numpy as np

fileDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = os.path.dirname(fileDirectory)

DATA_RECORD_NAME = os.path.join(parentDirectory, 'data/numbers_data_record.csv')
DATA_DIR = os.path.join(parentDirectory, 'data/')   

def chunk_input_stream(input_stream, label_length):
    """
    It takes an input stream and a label length, and returns a generator that yields chunks of the input
    stream of size 150 or 75, depending on the label length.
    
    :param input_stream: the input stream of data
    :param label_length: the length of the label (2 or 4)
    :return: A generator that yields a chunk of the input stream.
    """
    
    while True:
        if label_length is 2:
            chunk = list(islice(input_stream, 150))
            if chunk:
                if (len(chunk) % 150 == 0):
                    yield chunk
                else:
                    pad = [0] * ((300 - (len(chunk) % 150)))
                    yield chunk + pad
        elif label_length is 4:
            chunk = list(islice(input_stream, 75))
            if chunk:
                if (len(chunk) % 75 == 0):
                    yield chunk
                else:
                    pad = [0] * ((300 - (len(chunk) % 75)))
                    yield chunk + pad
        else:
            return

# # how to use
# output2 = np.array(list(chunk_input_stream((i for i in range(300)), 2))).astype('float32')
# output4 = np.array(list(chunk_input_stream((i for i in range(300)), 4))).astype('float32')

# print(output2.shape)
# print(output4.shape)
