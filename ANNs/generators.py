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


def generator(X, y, batch_size, window_size, threashold=0.5, predict_early=0):
    batch_features = np.zeros((batch_size, window_size, 1))
    batch_v = np.zeros((batch_size, window_size, 1))
    batch_labels = np.zeros((batch_size, 2))

    while True:
        for i in range(batch_size):
            index = np.random.randint(predict_early, X.shape[0]-window_size)
            a = X[index:index+window_size]
            a -= np.mean(a, keepdims=True) #subtracting the mean to make training easier
            batch_features[i] = np.reshape(a, (a.shape[0], -1))
            val = y[index-predict_early:index+window_size-predict_early]
            ripple_overlap = np.sum(val, axis=0)[1]/val.shape[0]
            if ripple_overlap>threashold:
                batch_labels[i] = np.array([0, 1])
            else:
                batch_labels[i] = np.array([1, 0])
        yield batch_features, batch_labels

def decode(value, threshold=0.992):
    mask = value[:,1]>threshold
    y_pred_int = np.array(mask, dtype=int)
    return y_pred_int



def generator_old(X, y, batch_size, window_size, threashold=0.5, predict_early=0):
    batch_features = np.zeros((batch_size, window_size, 1))
    batch_v = np.zeros((batch_size, window_size, 1))
    batch_labels = np.zeros((batch_size, 2))

    while True:
        for i in range(batch_size):
            index = np.random.randint(predict_early, X.shape[0]-window_size)
            a = X[index:index+window_size]
            batch_features[i] = np.reshape(a, (a.shape[0], -1))
            val = y[index-predict_early:index+window_size-predict_early]
            # b = v[index:index+window_size]
            # batch_v[i] = np.reshape(b, (b.shape[0], -1))
            if np.sum(val[:,0,1])>0:
                # print(np.max(np.squeeze(batch_v[i])), np.squeeze(batch_v[i]).shape)
                batch_labels[i] = np.array([0, 1])
            else:
                batch_labels[i] = np.array([1, 0])
        yield batch_features, batch_labels
