import pandas as pd
import numpy as np
import wfdb
import ast
import h5py
from scipy.signal import butter, lfilter


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = r'path/to/ptbxl/data/'
sampling_rate=500

# load and convert annotation data
path_to_ptbxl_database = path + 'ptbxl_database.csv'

Y = pd.read_csv(path_to_ptbxl_database, index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Preprocess raw data
# Order 5 works well with ECG signals
fs = 500
nyq = 0.5 * fs
cutoff = 5
normal_cutoff = cutoff / nyq
b, a = butter(1, normal_cutoff, btype = 'low', analog = False)

data = X
y = lfilter(b, a, data)

bh, ah = butter(1, normal_cutoff, btype = 'high', analog = False)
y_new = lfilter(bh, ah, y)

# Create training set
ptbxl_train_20k = y_new[0:20000]
with h5py.File(r'/path/to/training/set/ptbxl_train_20k.h5', 'w') as hdf:
    hdf.create_dataset('tracings', data = ptbxl_train_20k)

# Create test set
ptbxl_test_2k = y_new[20000:]
with h5py.File(r'path/to/test/setptbxl_test_2k.h5', 'w') as hdf:
    hdf.create_dataset('tracings', data = ptbxl_test_2k)

# 
path_to_store_all_data = r'path/to/store/data/ptbxl_all_data.h5'
with h5py.File(path_to_store_all_data, 'r') as f:
    data = np.array(f['ptbxl_all_data'])

