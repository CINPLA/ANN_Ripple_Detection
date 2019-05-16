import scipy.io as sio
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def read_out_arrays(data):
    lfp = data['lfp'][0][0]
    run_speed = data['run_speed'][0][0].flatten()
    ripple_loc = data['rippleLocs'][0][0].flatten()
    return lfp, run_speed, ripple_loc


def generate_data_set_for_animal(data, animal, ):
    lfp, run_speed, ripple_time = read_out_arrays(data[animal])

    label = np.zeros_like(lfp)
    label[ripple_time] = 1

    x = lfp
    y = label

    return x, y, run_speed


def generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat'):
    data = sio.loadmat(data_path)

    for key in data.keys():
        if key.startswith("m"):
            print('Generating', str(key))
            directory = os.path.join('..', 'data', 'processed_data', key)
            if not os.path.exists(directory):
                os.makedirs(directory)
            X_serial, y_serial, run_speed = generate_data_set_for_animal(data, key, )
            np.save(os.path.join(directory, 'X.npy'), X_serial)
            np.save(os.path.join(directory, 'y.npy'), y_serial)
            np.save(os.path.join(directory, 'meta.npy'), run_speed)


if __name__ == '__main__':
    generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat')
