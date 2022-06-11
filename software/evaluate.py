import numpy as np
import tensorflow.keras.backend as K
from Levenshtein import distance, editops, opcodes

def print_accuracy(predicted, truth, num_key):
    count = np.zeros((num_key, num_key + 2))

    for i in range(len(predicted)):
        y_hat = predicted[i]
        y = truth[i]
        y_pos = 0
        y_hat_pos = 0
        for op in opcodes(y_hat, y):
            optype = op[0]
            if optype != 'delete':
                count[int(y[y_pos]), num_key + 1] += 1
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
                count[int(y_hat[y_hat_pos]), num_key] += 1
                y_hat_pos += 1

    print('Count Table:')
    s = '     '
    for i in range(num_key):
        s += '%3d ' % i
    s += ' fa  all'
    print(s)
    for i in range(num_key):
        s = '%3d :' % i
        for j in range(num_key + 2):
            s += '%3d ' % count[i][j]
        print(s)

    print('Statistics:')
    print('label: all | right | wrong | miss | false positive')
    count_sum = {'all': 0, 'right': 0, 'wrong': 0, 'miss': 0, 'fp': 0}
    for c in range(num_key):
        count_wrong = np.sum(count[c, :num_key]) - count[c, c]
        count_miss = count[c, num_key + 1] - count[c, c] - count_wrong
        print('%5d: %3d | %5d | %5d | %4d | %11d' % (c, count[c, num_key + 1], count[c, c], count_wrong, count_miss, count[c, num_key]))
        count_sum['all'] += count[c][num_key + 1]
        count_sum['right'] += count[c, c]
        count_sum['wrong'] += count_wrong
        count_sum['miss'] += count_miss
        count_sum['fp'] += count[c, num_key]
    print('all, [%d, %d, %d, %d, %d]' % (count_sum['all'], count_sum['right'], count_sum['wrong'], count_sum['miss'], count_sum['fp']))
    print('accuracy: %.2f%%' % (100.0 * count_sum['right'] / count_sum['all']))

    return count

def model_accuracy(model, data, labels, input_length, label_length):
    loss, outputs = model.predict(data)
    y_pred = K.get_value(K.ctc_decode(outputs, input_length)[0][0])
    edit_distance = 0
    all_y = []
    all_y_hat = []
    for i in range(y_pred.shape[0]):
        y_pred_instance = y_pred[i][y_pred[i] != -1].astype(int)
        y_true_instance = labels[i][:label_length[i]].astype(int)
        # print(y_true_instance); print('==>'); print(y_pred_instance); print()
        all_y.append(y_true_instance)
        all_y_hat.append(y_pred_instance)
        edit_distance += distance(y_true_instance, y_pred_instance)
    num = sum(label_length)
    print('%.2f%% (error = %d, all = %d)' % (100 - 100 * edit_distance / num, edit_distance, num))
    count = print_accuracy(all_y_hat, all_y, 10)
    return 100 - 100 * edit_distance / num

