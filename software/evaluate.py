import numpy as np
import tensorflow.keras.backend as K
from Levenshtein import distance, editops, opcodes

def print_accuracy(y_pred, y_true):
    """
    It prints the accuracy of the predictions
    
    :param y_pred: the predicted labels
    :param y_true: the true labels
    :return: The count of the number of times each label was predicted correctly, incorrectly, missed,
    or falsely predicted.
    """
    count = np.zeros((10, 10 + 2))
    for i in range(len(y_pred)):
        y_hat = y_pred[i]
        y = y_true[i]
        y_pos = 0
        y_hat_pos = 0
        for op in opcodes(y_hat, y):
            optype = op[0]
            if optype != 'delete':
                count[int(y[y_pos]), 11] += 1
            if optype == 'equal':
                count[int(y[y_pos]), int(y[y_pos])] += 1
                y_pos += 1
                y_hat_pos += 1
            elif optype == 'replace':
                count[int(y[y_pos]), int(y_hat[y_hat_pos])] += 1
                y_pos += 1
                y_hat_pos += 1
            elif optype == 'insert':
                y_pos += 1
            else:
                count[int(y_hat[y_hat_pos]), 10] += 1
                y_hat_pos += 1

    print('Statistics:')
    print('label: all | right | wrong | miss | false positive')
    count_sum = {'all': 0, 'right': 0, 'wrong': 0, 'miss': 0, 'fp': 0}
    for c in range(10):
        count_wrong = np.sum(count[c, :10]) - count[c, c]
        count_miss = count[c, 11] - count[c, c] - count_wrong
        print('%5d: %3d | %5d | %5d | %4d | %11d' % (c, count[c, 11], count[c, c], count_wrong, count_miss, count[c, 10]))
        count_sum['all'] += count[c][11]
        count_sum['right'] += count[c, c]
        count_sum['wrong'] += count_wrong
        count_sum['miss'] += count_miss
        count_sum['fp'] += count[c, 10]
    print('all, [%d, %d, %d, %d, %d]' % (count_sum['all'], count_sum['right'], count_sum['wrong'], count_sum['miss'], count_sum['fp']))
    print('accuracy: %.2f%%' % (100.0 * count_sum['right'] / count_sum['all']))

    return count

def model_accuracy(model, data, labels, input_length, label_length):
    """
    It takes a model, some data, and some labels, and returns the accuracy of the model on that data
    
    :param model: the model to be evaluated
    :param data: the input data
    :param labels: the labels of the data
    :param input_length: the length of the input sequence
    :param label_length: the length of the label for each instance
    :return: The accuracy of the model
    """
    loss, outputs = model.predict(data)
    y_pred = K.get_value(K.ctc_decode(outputs, input_length)[0][0])
    edit_distance = 0
    y_trues = []
    y_preds = []
    for i in range(y_pred.shape[0]):
        y_pred_instance = y_pred[i][y_pred[i] != -1].astype(int)
        y_true_instance = labels[i][:label_length[i]].astype(int)
        y_trues.append(y_true_instance)
        y_preds.append(y_pred_instance)
        edit_distance += distance(y_true_instance, y_pred_instance)
    num = sum(label_length)
    print('%.2f%% (error = %d, all = %d)' % (100 - 100 * edit_distance / num, edit_distance, num))
    count = print_accuracy(y_preds, y_trues)
    return 100 - 100 * edit_distance / num

