import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU, Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam

start_time = time.time()

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

BATCH_SIZE = 220
N_EPOCHS = 20

def ctc_lambda_func(args):
    """
    It takes the output of the network, the labels, and the input length as arguments and returns the
    CTC loss
    :param args: A list of tensors
    :return: The cost of the batch.
    """
    y_pred, labels, input_length, label_length = args
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def create_model(feature_size, num_classes):
    """
    It creates a model with an input layer, a GRU layer, and a softmax layer
    
    :param feature_size: The number of features in the input data
    :param num_classes: The number of classes in the dataset
    :return: The model is being returned.
    """
    print('\nCreating Model....')
    input_data = Input(name='input', shape=(None, feature_size), dtype='float32')
    labels = Input(name='labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # GRU layer
    gru_layer = GRU(32, return_sequences=True, name='gru-layer')(input_data)

    # Softmax layer
    outputs = Dense(num_classes + 1, activation='softmax', name='softmax')(gru_layer)

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([outputs, labels, input_length, label_length])

    optimizer = Adam()

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    model.compile(
        loss={'ctc': lambda y_true, y_pred: y_pred},
        optimizer=optimizer
    )
    
    print(f'\nModel Created it took: {timedelta(seconds = (time.time() - start_time))}. Here is the summary:')
    
    model.summary()
    return model

def train(model, data, labels, input_length, label_length, data_val, labels_val, val_data_length, val_label_length, callbacks):
    """
    The function takes in the model, training data, training labels, input length, label length,
    validation data, validation labels, validation input length, validation label length, and callbacks.
    
    
    It then trains the model and returns the trained model.
    
    :param model: The model to train
    :param data: The input data
    :param labels: The labels for the data
    :param input_length: The length of the input sequence
    :param label_length: The length of the label for each input
    :param data_val: validation data
    :param labels_val: The validation labels
    :param val_data_length: The length of the validation data
    :param val_label_length: The length of the labels for the validation data
    :param callbacks: A list of callbacks to apply during training
    :return: The model is being returned.
    """
    inputs = {
        'input': data,
        'labels': labels,
        'input_length': input_length,
        'label_length': label_length
    }
    outputs = {'ctc': np.zeros([data.shape[0]])}

    validate_inputs = {
        'input': data_val,
        'labels': labels_val,
        'input_length': val_data_length,
        'label_length': val_label_length,
    }
    validate_outputs = {'ctc': np.zeros([data_val.shape[0]])}

    # Training
    print('\nTraining Model\n')
    model.fit(
        inputs, outputs,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=(validate_inputs, validate_outputs),
        callbacks=callbacks
    )
    
    print(f'\nModel trained, it took: {timedelta(seconds = (time.time() - start_time))}')
    
    return model

def test(model, data_test, labels_test, test_data_length, test_label_length):
    """
    This function takes in the model, the test data, the test labels, the test data length, and the test
    label length. It then creates a dictionary of the test inputs and outputs. The test outputs are set
    to an array of zeros with the same length as the test data. The model is then evaluated using the
    test inputs and outputs. The results are then printed
    
    :param model: The model we're training
    :param data_test: the test data
    :param labels_test: the labels for the test data
    :param test_data_length: The length of the test data
    :param test_label_length: The length of the labels for the test data
    :return: The model is being returned.
    """
    test_inputs = {
        'input': data_test,
        'labels': labels_test,
        'input_length': test_data_length,
        'label_length': test_label_length
    }
    test_outputs = {'ctc': np.zeros([data_test.shape[0]])}
    
    # Evaluate the model
    print('\nEvaluating Model...\n')
    test_loss = model.evaluate(test_inputs, test_outputs, batch_size = 440)
    
    print('\ntest loss, test acc:', test_loss)
    outputs = model.predict(test_inputs)
    print(outputs)

    return model 


def save(model):
    """
    **The save() function saves the model to a single HDF5 file which will contain:**
    
    - the architecture of the model, allowing to re-create the model
    - the weights of the model
    - the training configuration (loss, optimizer)
    - the state of the optimizer, allowing to resume training exactly where you left off
    
    :param model: The model to be saved
    """
    model.save('saved_model.h5')
    print('\nModel has been saved')
