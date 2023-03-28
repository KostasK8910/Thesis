import pandas as pd
import numpy as np
import wfdb
import ast
import h5py
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = r'path/to/ptbxl/dataset'
sampling_rate=500

# load and convert annotation data
path_to_ptbxl_database = path + 'ptbxl_database.csv'

Y = pd.read_csv(path_to_ptbxl_database, index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
#search_dict = {'AFIB', 'STATCH', 'SBRAD', '1AVB', 'CRBBB', 'LBBBB'}

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a

# A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a

def final_filter(data, fs, order=5):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order = order)
    y = lfilter(d, c, x)
    return y


## Order of five works well with ECG signals
cutoff_high = 0.5
cutoff_low = 50
order = 5
fs = 500

prepr_ptbxl_all = final_filter(X, fs, order)
prepr_ptbxl_20k = prepr_ptbxl_all[0:20000, :, :]
prepr_ptbxl_2k = prepr_ptbxl_all[20000:, :, :]

# Create preprocessed training set
path_to_20k = r'path/to/preprocessed/training/set/prepr_ptbxl_20k.hdf5'
with h5py.File(path_to_20k, 'w') as hdf:
    hdf.create_dataset('tracings', data = prepr_ptbxl_20k)

# Create preprocessed test set
path_to_2k = r'path/to/preprocessed/test/set/prepr_ptbxl_2k.hdf5'
with h5py.File(path_to_2k, 'w') as hdf:
    hdf.create_dataset('tracings', data = prepr_ptbxl_2k)





