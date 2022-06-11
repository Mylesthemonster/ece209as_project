import time
from datetime import timedelta
from urllib import response

import numpy as np
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU, Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam

import time
from datetime import timedelta
from tensorflow.keras.backend import ctc_decode, get_value, function, learning_phase
from evaluate import model_accuracy
import tensorflow.keras.backend as K
from Levenshtein import distance
import matplotlib.pyplot as plt

start_time = time.time()

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

BATCH_SIZE = 220
N_EPOCHS = 30
WEIGHTS_PATH = 'model.h5'

def ctc_lambda_func(args):
    """
    It takes the output of the network, the labels, and the input length as arguments and returns the
    CTC loss
    :param args: A list of tensors
    :return: The cost of the batch.
    """
    y_pred, labels, input_length, label_length = args
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_decode_lambda_func(args):
    y_pred, input_length, labels, label_length = args
    decoded = keras.backend.ctc_decode(y_pred, K.squeeze(input_length, axis=-1))[0][0]
    s1 = tf.sparse.from_dense(decoded[decoded != -1])
    s2 = tf.sparse.from_dense(tf.cast(labels, tf.int64))
    return tf.reduce_mean(tf.edit_distance(s1, s2))

def accuracy(y_pred, y_true):
    acc = y_pred[1]
    return acc


def create_model(feature_size, num_classes, gru_size=64):
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

    gru = GRU(gru_size, return_sequences=True, name='gru1')(input_data)
    dense1 = Dense(100, name='dense1')(gru)
    # dense2 = Dense(200, name='dense2')(dense1)
    outputs = Dense(num_classes + 1, activation='softmax', name='output')(dense1)

    loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
    # acc_out = Lambda(
    #     ctc_decode_lambda_func,
    #     name='acc')([outputs, input_length, labels, label_length])
    
    test_model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=[loss, outputs]
                  )

    optimizer = Adam()

    train_model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=[loss]
                  )

    train_model.compile(
        loss={'ctc': lambda y_true, y_pred: y_pred},
        # metrics=[accuracy],
        optimizer=optimizer
    )

    train_model.summary()
    return train_model, test_model

def train(model, data, labels, input_length, label_length, data_val, labels_val, val_data_length, val_label_length, plot=False):
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
    history = model.fit(
        inputs, outputs,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=(validate_inputs, validate_outputs)
    )

    print(f'\nModel trained, it took: {timedelta(seconds = (time.time() - start_time))}')
    
    if plot:
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
    
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
    
    return model_accuracy(model, test_inputs, labels_test, test_data_length, test_label_length)


FINE_TUNE_BATCH_SIZE = 10

def test_and_update(model, data_test, labels_test, test_data_length, test_label_length):
    """
    The test_and_update() function tests the model accuracy and also fine tunes it with new data
    """
    test_inputs = {
        'input': data_test,
        'labels': labels_test,
        'input_length': test_data_length,
        'label_length': test_label_length
    }
    test_outputs = {'softmax': np.zeros([labels_test.shape[0]]),
    'ctc': np.zeros([data_test.shape[0]])}
    
    # new_model = Model(inputs=model.input, outputs=model.get_layer("softmax").output)
    # model_accuracy(model, test_inputs, labels_test, test_data_length, test_label_length)

    # print(labels_test.shape)
    
    # # for layer in model.layers:
    # #     layer.trainable = False
    # # model.get_layer('softmax').trainable = True
    # optimizer = Adam()

    # model.compile(
    #     loss={'ctc': lambda y_true, y_pred: y_pred},
    #     optimizer=optimizer
    # )
    # model.fit(
    #     test_inputs, test_outputs,
    #     batch_size=BATCH_SIZE,
    #     epochs=4,
    # )

    # model_accuracy(model, test_inputs, labels_test, test_data_length, test_label_length)
    # print('\ntest loss, test acc:', test_loss)
    
    # Model Prediction
    output = model.predict(test_inputs)
    
    for i in range(len(output)-1):
        x = test_inputs['labels'][i]
        y = output[i]
        print(f'\nCase {i+1} of {len(output)}:\nPredicted output is : {y}\nCorrect output is: {x}')
        
        while True:
            try:
                response = input("\nDo they match (y/n): ")
                if response.lower() == 'y':
                    print('\nThank you for response')
                    break
                
                elif response.lower() == 'n':
                    print('\nError Recorded')
                    break
                else:
                    print('\nResponse not valid please try again')
                    
            except ValueError:
                print ('\nCongrats you broke it')

    return model 


def save_model(model, path):
    """
    **The save_model() function saves the model to a single HDF5 file which will contain:**
    
    - the architecture of the model, allowing to re-create the model
    - the weights of the model
    - the training configuration (loss, optimizer)
    - the state of the optimizer, allowing to resume training exactly where you left off
    
    :param model: The model to be saved
    :param path: The path to which to save the model
    """
    model.save_weights(path)
    print("\nModel has been saved")

def load_model(model, path):

    """
    **The load_model() function loads a single HDF5 file which will contain the weights of the model **

    :param model: The model to be loaded to
    :param path: The path to which to load the model
    """
    model.load_weights(path, by_name=True)
