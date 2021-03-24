from concurrent.futures.thread import ThreadPoolExecutor

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
a = tf.constant([2.0, 3.0, 4.0])
b = tf.constant([2.0, 3.0, 4.0])
c = np.array([1,2,3,4,5,6,7,8])
b= tf.transpose(b)
d = K.dot(tf.reshape(a, (1,3)), tf.reshape(b, (3,1)))
# print(K.arange(0,100,50))

def my_move_average_overlap_tf(data, win_size=200, overlap=100, axis=-1):
    if len(data.shape) == 1:
        data = data.reshape((1, -1))
    ret = K.cumsum(data, axis=axis)
    result = (ret[:, win_size:] - ret[:, :-win_size]) / win_size
    index = np.arange(0, result.shape[1], overlap) + 1
    return result[:, index]

def my_move_average_overlap(data, win_size=200, overlap=100, axis=-1):
    if len(data.shape) == 1:
        data = data.reshape((1, -1))

    ret = np.cumsum(data, axis=axis)

    # ret[:, win_size:] = ret[:, win_size:] - ret[:, :-win_size]
    # result = ret[:, win_size - 1:] / win_size

    result = (ret[:, win_size:] - ret[:, :-win_size]) / win_size

    index = np.arange(0, result.shape[1], overlap) + 1
    return result[:, index]

v = my_move_average_overlap_tf(c.reshape(1,-1), win_size=2, overlap=2, axis=-1)
print(v)