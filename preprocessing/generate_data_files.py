import scipy.io as sio
import numpy as np
import os
import warnings

from ripple_detection import Karlsson_ripple_detector, Kay_ripple_detector
from ripple_detection.simulate import simulate_time

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_out_arrays(data):
    lfp = data['lfp'][0][0]
    run_speed = data['run_speed'][0][0]
    ripple_loc = data['rippleLocs'][0][0].flatten()
    min_val = min(lfp.shape[0], run_speed.shape[0])
    return lfp[:min_val,:], run_speed[:min_val,:], ripple_loc


def generate_data_set_for_animal(data, animal, ):
    lfp, speed, ripple_time = read_out_arrays(data[animal])

    sf = 2.5e3
    time = simulate_time(lfp.shape[0], sf)
    Kay_ripple_times = Kay_ripple_detector(time, lfp, speed.flatten(), sf)

    label = np.zeros_like(time)
    for i in range(Kay_ripple_times.shape[0]):
        start_index = int(np.argwhere(time==np.array(Kay_ripple_times)[i,0]))
        end_index = int(np.argwhere(time==np.array(Kay_ripple_times)[i,1]))
        label[start_index:end_index] = 1
    # label = np.zeros_like(lfp)
    # label[ripple_time] = 1

    x = lfp
    y = label

    return x, y, speed, time, np.array(Kay_ripple_times)


def generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat'):
    data = sio.loadmat(data_path)

    for key in data.keys():
        if key.startswith("m"):
            print('Generating', str(key))
            directory = os.path.join('..', 'data', 'processed_data', key)
            if not os.path.exists(directory):
                os.makedirs(directory)
            X_serial, y_serial, speed, time, ripple_periods = generate_data_set_for_animal(data, key, )
            np.save(os.path.join(directory, 'X.npy'), X_serial)
            np.save(os.path.join(directory, 'y.npy'), y_serial)
            np.save(os.path.join(directory, 'speed.npy'), speed)
            np.save(os.path.join(directory, 'time.npy'), time)
            np.save(os.path.join(directory, 'ripple_periods.npy'), ripple_periods)


if __name__ == '__main__':
    generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat')
