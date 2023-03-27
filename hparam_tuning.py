from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from model import get_model
from datasets import ECGSequence
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import gc
from tensorflow.python.keras.metrics import binary_accuracy

kernel_sizes = [3, 7, 11, 15]
dropout_keep_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
optimizers = ['adam', 'sgd']

val_split = 0.02
dataset_name = 'tracings'
epochs_per_file = 10
# Optimization settings
loss = 'binary_crossentropy'
lr = 0.001
batch_size = 64
folderpath = r'/path/to/data/directory'
path_to_save_models = folderpath + r'/hparam_models'

def train_model(kernel_size, dropout_keep_prob, optimizer, run_num):
    kernel_size = kernel_size
    dropout_keep_prob = dropout_keep_prob
    optimizer = optimizer
    run_num = run_num

    for i in range(18):
        path_to_hdf5 = folderpath + '/exams_part' + str(i) + '.hdf5'
        path_to_csv = folderpath + '/exam_part' + str(i) + '.csv'
        train_seq, valid_seq = ECGSequence.get_train_and_val(path_to_hdf5, dataset_name, 
                                                             path_to_csv, batch_size, val_split)

        # If you are continuing an interrupted section, uncomment line bellow:
        # model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
        model = get_model(train_seq.n_classes)
        model.compile(loss=loss, optimizer=optimizer)
    
        callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]
    
        # Create log
        callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
        # Save the BEST and LAST model
        callbacks += [ModelCheckpoint('./backup_model_last.hdf5'),
                  ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]
        # Train neural network
        if i == 0:
            history = model.fit(train_seq,
                        epochs=epochs_per_file,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)
            model.save(path_to_save_models + '/run' + str(run_num) +'/model_part0.hdf5')

        else:
            previous_model = path_to_save_models + '/run' + str(run_num) + '/model_part' + str(i-1) +'.hdf5'
            model = load_model(previous_model)
            history = model.fit(train_seq,
                        epochs=epochs_per_file*i + epochs_per_file,
                        initial_epoch=epochs_per_file*i,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)
            model.save(path_to_save_models + '/run' + str(run_num) + '/model_part' + str(i) + '.hdf5')
        binary_accuracy = model.evaluate(valid_seq.x, valid_seq.y)
        del train_seq.x
        del train_seq.y
        del valid_seq.x
        del valid_seq.y
        gc.collect()   # remove data to free up ram

        # remove previously saved model to free up storage space in disk 
        if i>0:
            path = path_to_save_models + '/run' + str(run_num) + '/model_part' + str(i-1) + '.hdf5'
            os.remove(path)
    return binary_accuracy

session_num = 30
bin_acc = []

for kernel_size in kernel_sizes:
    for dropout_keep_prob in dropout_keep_probs:
        for optimizer in optimizers:
            kernel_size = kernel_size
            dropout_keep_prob = dropout_keep_prob
            optimizer = optimizer
            
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print('kernel_size = ', kernel_size, ', dropout_keep_prob = ', dropout_keep_prob, ', optimizer = ', optimizer)
            bin_accuracy = train_model(kernel_size = kernel_size, dropout_keep_prob = dropout_keep_prob, optimizer = optimizer, run_num = session_num)
            print('binary accuracy of run ', session_num, ' is ', bin_accuracy)
            bin_acc.append([kernel_size, dropout_keep_prob, optimizer, bin_accuracy])
            session_num += 1

df = pd.DataFrame(bin_acc)
path_to_bin_acc = r'/path/to/save/results/results.csv'
df.to_csv(path_to_bin_acc, index = False)
  