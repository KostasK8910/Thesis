from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.models import load_model
from model import get_model
import pandas as pd
import argparse
from datasets import ECGSequence
import sys
import gc

val_split = 0.02
dataset_name = 'tracings'
epochs_per_file = 20
# Optimization settings
loss = 'binary_crossentropy'
lr = 0.001
batch_size = 32
opt = Adam(lr)
callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

folderpath = r'/path/to/data/directory'
path_to_save_models = r'/path/to/save/models'

for i in range(18):
    path_to_hdf5 = folderpath + '/exams_part' + str(i) + '.hdf5'
    path_to_csv = folderpath + '/exam_part' + str(i) + '.csv'
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        path_to_hdf5, dataset_name, path_to_csv, batch_size, val_split)

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(train_seq.n_classes)
    model.compile(loss=loss, optimizer=opt)

    # Create log
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.hdf5'),
                  ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]
    # Train neural network
    if i == 0:
        model.fit(train_seq,
                        epochs=epochs_per_file,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)
        model.save(path_to_save_models + '/model_part0.hdf5')
        del train_seq.x
        del train_seq.y
        del valid_seq.x
        del valid_seq.y
        gc.collect()

    else:
        previous_model = path_to_save_models + '/model_part' + str(i-1) +'.hdf5'
        next_model = load_model(previous_model)
        next_model.fit(train_seq,
                        epochs=epochs_per_file*i + epochs_per_file,
                        initial_epoch=epochs_per_file*i,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)
        del train_seq.x
        del train_seq.y
        del valid_seq.x
        del valid_seq.y
        gc.collect()
        next_model.save(path_to_save_models + '/model_part' + str(i) + '.hdf5')
