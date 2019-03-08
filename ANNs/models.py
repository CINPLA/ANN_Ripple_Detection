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


def save_model(model, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    t = time.strftime("%Y%m%d-%H:%M:%S")
    path = os.path.join(save_dir, t+'.h5')
    model.save(path)


def generator(X, y, batch_size, lookback, return_full_stream=False, threashold=0.5):
    batch_features = np.zeros((batch_size, lookback, 1))
    batch_labels = np.zeros((batch_size, 2))
    if return_full_stream:
        batch_labels_full = np.zeros((batch_size, lookback, 2))
    while True:
        for i in range(batch_size):
            index = np.random.randint(0, X.shape[0]-lookback)
            a = X[index:index+lookback]
            batch_features[i] = np.reshape(a, (a.shape[0], -1))
            val = y[index:index+lookback]
            if return_full_stream:
                batch_labels_full[i] = y[index:index+lookback]
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


def generate_model(input_shape, padding='same'):
    keras.backend.clear_session()

    input_layer = keras.layers.Input(shape=input_shape)

    conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
    conv1 = keras.layers.AveragePooling1D(pool_size=2)(conv1)

    conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
    conv2 = keras.layers.AveragePooling1D(pool_size=2)(conv2)

    conv3 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv2)
    conv3 = keras.layers.AveragePooling1D(pool_size=2)(conv3)

    flatten_layer = keras.layers.Flatten()(conv3)

    full_conencted = keras.layers.Dense(200)(flatten_layer)

    output_layer = keras.layers.Dense(units=2,activation='softmax')(full_conencted)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=keras.optimizers.Adam(lr=0.005),
                  metrics=['accuracy'],
                  )

    return model


def plot_performance(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


def decode(value, threshold=0.992):
    mask = value[:,1]>threshold
    y_pred_int = np.array(mask, dtype=int)
    return y_pred_int

def decode2(value, threshold=0.5):
    return np.argmax(value, axis=-1)


def make_accuracy_matrix_plot(model, validate_generator, ref='truth'):
    X_trial, y_trial, = next(validate_generator)
    res = model.predict(X_trial)

    print(y_trial)
    print(res)
    
    y = decode(y_trial)
    res = decode(res)

   
    plt.figure()
    plt.plot(y)
    plt.plot(res)
    plt.show()
    
    res_matrix = np.zeros((2,2))
    if ref=='truth':
        for i, true_label in enumerate(set(y)):
            for j, reco_label in enumerate(set(y)):
                mask_true = y==true_label
                mask_reco = res==reco_label
                res_matrix[i,j] = np.sum(mask_true*mask_reco) / np.sum(mask_true)
    if ref=='est':
        for i, true_label in enumerate(set(y)):
            for j, reco_label in enumerate(set(y)):
                mask_true = y==true_label
                mask_reco = res==reco_label
                res_matrix[i,j] = np.sum(mask_true*mask_reco) / np.sum(mask_reco)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(res_matrix, norm=mpl.colors.Normalize(vmin=0., vmax=1.))
    for i in range(res_matrix.shape[0]):
        for j in range(res_matrix.shape[0]):
            if res_matrix[i, j]>0.5:
                color = 'k'
            else:
                color = 'w'
            text = ax.text(j, i, np.round(res_matrix[i, j], decimals=3),
            ha='center', va='center', color=color)
    ax.set_xlabel('CNN class')
    ax.set_ylabel('manual class')
    ax.xaxis.set_ticks_position('top')
    # ax.colorbar(im)
    plt.savefig('plots/res.pdf')
    plt.show()
