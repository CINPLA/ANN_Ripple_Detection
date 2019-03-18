import scipy.io as sio
import numpy as np
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import datetime
import quantities as pq
from ripple_detection import Karlsson_ripple_detector, Kay_ripple_detector
from ripple_detection.simulate import simulate_time
import scipy.stats

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def read_out_arrays(data):
    lfp = data['lfp'][0][0]
    run_speed = data['run_speed'][0][0].flatten()
    ripple_loc = data['rippleLocs'][0][0].flatten()
    return lfp, run_speed, ripple_loc


def select_manual_scored_ripples(detector, ripple_time, time):
    ripple_mask = np.zeros_like(time)
    ripple_mask[ripple_time] = 1
    delete_mask = np.zeros(detector.shape[0], dtype=bool)
    for i, ripple in enumerate(detector.itertuples()):
        mask = np.logical_and(ripple.start_time<=time, ripple.end_time>=time)
        delete_mask[i] = np.max(ripple_mask[mask])==1
    return detector[delete_mask]


def generate_label_array(detector, time):
    y = np.zeros_like(time)
    for i, ripple in enumerate(detector.itertuples()):
        mask = (np.logical_and(ripple.start_time<=time, ripple.end_time>=time))
        y[mask] = 1
    return y


def generate_data_set_for_animal(data, animal, ):
    lfp, run_speed, ripple_time = read_out_arrays(data[animal])
    SAMPLING_FREQUENCY = 2.5e3
    n_samples = lfp.shape[0]
    time = simulate_time(n_samples, SAMPLING_FREQUENCY)
    speed = np.ones_like(time)

    Karlsson_ripple_times = Karlsson_ripple_detector(time, lfp, speed, SAMPLING_FREQUENCY)
    Kay_ripple_times = Kay_ripple_detector(time, lfp, speed, SAMPLING_FREQUENCY)

    validated_ripples = select_manual_scored_ripples(Kay_ripple_times, ripple_time, time)

    x = lfp
    y = generate_label_array(validated_ripples, time)

    return x, y


def generate_batch_data(X_serial, y_serial, scaling=100):
    series_length = 200
    n_series = int(np.floor(X_serial.shape[0] / series_length)) * scaling

    X = np.empty((n_series, series_length, X_serial.shape[-1]))
    y = np.empty((n_series, series_length, ))

    for i in range(n_series):
        start_index = np.random.randint(0, X_serial.shape[0]-series_length)
        X[i] = X_serial[start_index:start_index+series_length]
        y[i] = y_serial[start_index:start_index+series_length]

    return X, y


def generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat'):
    data = sio.loadmat(data_path)

    for key in data.keys():
        if key.startswith("m"):
            print('Generating', str(key))
            directory = os.path.join('..', 'data', 'processed_data', key)
            if not os.path.exists(directory):
                os.makedirs(directory)
            X_serial, y_serial = generate_data_set_for_animal(data, key, )
            X, y = generate_batch_data(X_serial, y_serial, scaling=10)
            np.save(os.path.join(directory, 'X_serial.npy'), X_serial)
            np.save(os.path.join(directory, 'y_serial.npy'), y_serial)
            # np.save(os.path.join(directory, 'X.npy'), X)
            # np.save(os.path.join(directory, 'y.npy'), y)

if __name__ == '__main__':
    generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat')
