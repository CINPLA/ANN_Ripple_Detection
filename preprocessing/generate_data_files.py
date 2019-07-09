import scipy.io as sio
import numpy as np
import scipy
import os
import glob
import warnings
from scipy import signal

from ripple_detection import Karlsson_ripple_detector, Kay_ripple_detector
from ripple_detection.simulate import simulate_time

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_out_arrays(data):
    lfp = data['lfp'][0][0]
    run_speed = data['run_speed'][0][0]
    ripple_loc = data['rippleLocs'][0][0].flatten()
    min_val = min(lfp.shape[0], run_speed.shape[0])
    return lfp[:min_val,:], run_speed[:min_val,:], ripple_loc


def generate_data_set_for_animal(data, animal, sf=2.5e3, q=1):
    lfp, speed, ripple_index = read_out_arrays(data[animal])

    time = simulate_time(lfp.shape[0], sf)
    ripple_times = time[ripple_index]

    lfp = scipy.signal.decimate(lfp.flatten(), q)
    # lfp = lfp.flatten()

    def perform_high_pass_filter(lfp, low_cut_frequency, high_cut_frequency, sf):
            wn = sf / 2.
            b, a = signal.butter(5, [low_cut_frequency/wn, high_cut_frequency/wn], 'bandpass')
            lfp = signal.filtfilt(b, a, lfp)
            return lfp

    lfp = perform_high_pass_filter(lfp, 1, 500, sf)

    lfp = lfp[:, np.newaxis]
    speed = scipy.signal.decimate(speed.flatten(), q)
    time = simulate_time(lfp.shape[0], sf/q)

    ripple_time_index_sparse = list()
    for t in ripple_times:
        ripple_time_index_sparse.append(np.argmin(np.abs(t-time)))


    Kay_ripple_times = Kay_ripple_detector(time, lfp, speed.flatten(), sf/q)
    label = np.zeros_like(time)
    ripple_index = list()
    for i in range(Kay_ripple_times.shape[0]):
        start_index = int(np.argwhere(time==np.array(Kay_ripple_times)[i,0]))
        end_index = int(np.argwhere(time==np.array(Kay_ripple_times)[i,1]))
        label[start_index:end_index] = 1
        ripple_index.append([start_index, end_index])

    res = dict()
    res['sf'] = sf/q
    res['X'] = lfp
    res['y'] = label
    res['speed'] = speed
    res['time'] = time
    res['ripple_times'] = ripple_times
    res['ripple_time_index'] = np.array(ripple_time_index_sparse)
    res['ripple_periods'] = np.array(Kay_ripple_times)
    res['ripple_index'] = np.array(ripple_index)

    return res


def generate_individual_objects(data, window_size=2.):
    window_size = window_size * data['sf']
    half_wdith = int(window_size/2)

    batch = list()
    label = list()

    for i in range(data['ripple_index'].shape[0]):
        mask = np.zeros_like(data['time'])
        full_ripple_times = np.zeros_like(mask)
        full_ripple_times[data['ripple_time_index']] = 1
        ripple_center = int(np.round((data['ripple_index'][i,1]+data['ripple_index'][i,0])/2))
        mask[ripple_center-half_wdith:ripple_center+half_wdith] = 1
        mask = np.array(mask, dtype=bool)
        batch.append(np.array(data['X'][mask]))
        overlap = np.sum(full_ripple_times * mask), np.sum(mask)
        if overlap[0]>0:
            label.append(1)
        else:
            label.append(0)
    return np.array(batch), np.array(label)


def merge_all_data():

    directory = '../data/merged_data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    X_all = list()
    y_all = list()

    file_list = glob.glob('../data/processed_data/*')

    for f in file_list[:-1]:
        name = f.split('/')[-1]
        X = np.load(os.path.join(f, 'X.npy'))
        y = np.load(os.path.join(f, 'y.npy'))
        X_all.append(X)
        y_all.append(y)
        # print(file['X'].shape)


    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    print(X_all.shape, y_all.shape)

    np.save(os.path.join(directory,'X.npy'), X_all)
    np.save(os.path.join(directory,'y.npy'), y_all)


def generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat'):
    data = sio.loadmat(data_path)

    directory = os.path.join('..', 'data', 'processed_data')
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key in data.keys():
        if key.startswith("m"):
            print('Generating', str(key))
            directory = os.path.join('..', 'data', 'processed_data', key)
            if not os.path.exists(directory):
                os.makedirs(directory)
            res = generate_data_set_for_animal(data, key, sf=2.5e3, q=1)
            X, y = generate_individual_objects(res, window_size=0.5)
            np.save(os.path.join(directory, 'all.npy'), res)
            np.save(os.path.join(directory, 'X.npy'), X)
            np.save(os.path.join(directory, 'y.npy'), y)

    merge_all_data()


if __name__ == '__main__':
    generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat')
