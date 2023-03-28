import pandas as pd
import numpy as np
import wfdb
import ast
import h5py
import matplotlib.pyplot as plt


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = r'path/to/ptbxl/'
sampling_rate=500

# load and convert annotation data
path_to_ptbxl_database = path + 'ptbxl_database.csv'

Y = pd.read_csv(path_to_ptbxl_database, index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

raw_ptbxl_20k = X[0:20000, :, :]
raw_ptbxl_2k = X[20000:, :, :]
path_to_20k =  r'path/to/store/raw/training/data/raw_ptbxl_20k.hdf5'
path_to_2k =  r'path/to/store/raw/test/data/raw_ptbxl_2k.hdf5'

# Create training set of raw ptbxl data
with h5py.File(path_to_20k, 'w') as f:
    f.create_dataset('tracings', data = raw_ptbxl_20k)

# Create test set of raw ptbxl data    
with h5py.File(path_to_2k, 'w') as g:
    g.create_dataset('tracings', data = raw_ptbxl_2k)

