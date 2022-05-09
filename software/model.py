from cProfile import label
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Lambda, GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as load

BATCH_SIZE = 220
N_EPOCHS = 20

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def create_model(feature_size, num_classes):
    input_data = Input(name='input', shape=(None, feature_size), dtype='float32')
    labels = Input(name='labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # GRU layer
    gru1 = GRU(32, return_sequences=True, name='gru')(input_data)

    # Softmax layer
    outputs = Dense(num_classes + 1, activation='softmax', name='softmax')(gru1)

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([outputs, labels, input_length, label_length])

    optimizer = Adam()

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    model.summary()
    return model

def train(model, data, labels, input_length, label_length, callbacks):
    inputs = {
        'input': data,
        'labels': labels,
        'input_length': input_length,
        'label_length': label_length
    }
    outputs = {'ctc': np.zeros([data.shape[0]])}

    # validate_inputs = {
    #     'input': test_data,
    #     'labels': test_labels,
    #     'input_length': test_input_length,
    #     'label_length': test_label_length,
    # }
    # validate_outputs = {'ctc': np.zeros([test_data.shape[0]])}

    # training
    model.fit(
        inputs, outputs,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        # validation_data=(validate_inputs, validate_outputs),
        callbacks=callbacks
    )
    return model