import scipy.io as sio
import numpy as np
import scipy
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


def generate_data_set_for_animal(data, animal, sf=2.5e3, q=3):
    lfp, speed, ripple_times = read_out_arrays(data[animal])

    time = simulate_time(lfp.shape[0], sf)
    Kay_ripple_times = Kay_ripple_detector(time, lfp, speed.flatten(), sf, speed_threshold=1000.0)

    label = np.zeros_like(time)
    for i in range(Kay_ripple_times.shape[0]):
        start_index = int(np.argwhere(time==np.array(Kay_ripple_times)[i,0]))
        end_index = int(np.argwhere(time==np.array(Kay_ripple_times)[i,1]))
        label[start_index:end_index] = 1


    lfp2 = scipy.signal.decimate(lfp.flatten(), q)
    lfp2 = lfp2[:, np.newaxis]
    speed2 = scipy.signal.decimate(speed.flatten(), q)
    time2 = simulate_time(lfp2.shape[0], sf/q)
    Kay_ripple_times2 = Kay_ripple_detector(time2, lfp2, speed2.flatten(), sf/q)

    label2 = np.zeros_like(time2)
    for i in range(Kay_ripple_times2.shape[0]):
        start_index = int(np.argwhere(time2==np.array(Kay_ripple_times2)[i,0]))
        end_index = int(np.argwhere(time2==np.array(Kay_ripple_times2)[i,1]))
        label2[start_index:end_index] = 1

    res = dict()
    res['X_unscaled'] = lfp
    res['y_unscaled'] = label
    res['speed_unscaled'] = speed
    res['time_unscaled'] = time
    res['ripple_times_unscaled'] = ripple_times
    res['ripple_periods_unscaled'] = np.array(Kay_ripple_times)

    res['X'] = lfp2
    res['y'] = label2
    res['speed'] = speed2
    res['time'] = time2
    res['ripple_times'] = ripple_times
    res['ripple_periods'] = np.array(Kay_ripple_times2)

    return res


def generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat'):
    data = sio.loadmat(data_path)

    for key in data.keys():
        if key.startswith("m"):
            print('Generating', str(key))
            directory = os.path.join('..', 'data', 'processed_data', key)
            if not os.path.exists(directory):
                os.makedirs(directory)
            res = generate_data_set_for_animal(data, key, )
            np.save(os.path.join(directory, 'all.npy'), res)
            np.save(os.path.join(directory, 'X.npy'), res['X'])
            np.save(os.path.join(directory, 'y.npy'), res['y'])
            np.save(os.path.join(directory, 'speed.npy'), res['speed'])
            np.save(os.path.join(directory, 'time.npy'), res['time'])
            np.save(os.path.join(directory, 'ripple_periods.npy'), res['ripple_periods'])
            np.save(os.path.join(directory, 'ripple_times.npy'), res['ripple_times'])


if __name__ == '__main__':
    generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat')
