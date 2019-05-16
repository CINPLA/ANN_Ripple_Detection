import tensorflow as tf
import numpy as np
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import matplotlib as mpl

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model


def generator(X, y, batch_size, lookback, return_full_stream=False, threashold=0.5, predict_early=0):
    batch_features = np.zeros((batch_size, lookback, 1))
    batch_labels = np.zeros((batch_size, 2))
    if return_full_stream:
        batch_labels_full = np.zeros((batch_size, lookback, 2))
    while True:
        for i in range(batch_size):
            index = np.random.randint(predict_early, X.shape[0]-lookback)
            a = X[index:index+lookback]
            batch_features[i] = np.reshape(a, (a.shape[0], -1))
            val = y[index-predict_early:index+lookback-predict_early]
            if return_full_stream:
                batch_labels_full[i] = y[index-predict_early:index+lookback-predict_early]
            rate = np.sum(val, axis=0)
            ratio = rate[1] / float(val.shape[0])
            if ratio>=threashold:
                batch_labels[i] = np.array([0, 1])
            else:
                batch_labels[i] = np.array([1, 0])
        if return_full_stream:
            yield batch_features, batch_labels, batch_labels_full
        else:
            yield batch_features, batch_labels

def decode(value, threshold=0.992):
    mask = value[:,1]>threshold
    y_pred_int = np.array(mask, dtype=int)
    return y_pred_int


def decode2(value, threshold=0.5):
    return np.argmax(value, axis=-1)
