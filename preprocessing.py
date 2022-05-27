import glob
import os
import pickle
import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from datetime import datetime
LON_SIZE = 72
LAT_SIZE = 36

import tensorflow as tf
from tensorflow import keras

BATCH_SIZE = 512
EPOCHS = 135
LEARNING_RATE = 0.005
NUM_FEATURES = 1*3

def get_cnn_model():
    inputs = keras.Input((LON_SIZE, LAT_SIZE, NUM_FEATURES))
    conv1 = keras.layers.Conv2D(64, [3,3], activation='tanh', padding='same', strides=1,kernel_initializer='glorot_normal')(inputs)
    pool1 = keras.layers.MaxPooling2D((2,2), strides=2, padding='same')(conv1)
    conv2 = keras.layers.Conv2D(96, [5,5], activation='tanh', padding='same', strides=1)(pool1)
    pool2 = keras.layers.MaxPooling2D((2,2), strides=2, padding='same')(conv2)
    conv3 = keras.layers.Conv2D(64, [5,5], activation='tanh', padding='same', strides=1)(pool2)
    flat = keras.layers.Flatten()(conv3)
    dense1 = keras.layers.Dense(512, activation='tanh')(flat)
    outputs = keras.layers.Dense(1, activation=None)(dense1)
    cnn_model = keras.Model(inputs=inputs, outputs=outputs) 
    cnn_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE), loss='mse')
    return cnn_model


# # Utility for running experiments.
def run_experiment(ldmn, model_number=0):
    indir = f"/media/cmlws/Data2/jsp/LMEdata/{ldmn}/"
    chl_train_x = np.load(f"{indir}/chl_tr_x.npy")
    chl_valid_x = np.load(f"{indir}/chl_val_x.npy")
    chl_test_x = np.load(f"{indir}/chl_test_x.npy")
    
    sst_train_x = np.load(f"{indir}/sst_tr_x.npy")
    sst_valid_x = np.load(f"{indir}/sst_val_x.npy")
    sst_test_x = np.load(f"{indir}/sst_test_x.npy")

    train_x = np.load(f"{indir}/tr_x.npy")
    valid_x = np.load(f"{indir}/val_x.npy")
    test_x = np.load(f"{indir}/test_x.npy")

    train_y = np.load(f"{indir}/tr_y.npy")
    valid_y = np.load(f"{indir}/val_y.npy")


    outdir = f"../output/xrchl_t6vRe_hp/models/{ldmn}"
    os.makedirs(outdir, exist_ok=True)       
#텐서보드 로그 저장 디렉토리 
    logdir= outdir+"/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(logdir, exist_ok=True)

    # 아웃풋 저장 디렉토리 
    os.makedirs(outdir, exist_ok=True)
    filepath = outdir
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath+'/model'+str(model_number)+'.h5',
            monitor='val_loss',
            save_best_only=True, 
            verbose=1
            )
    ]
    cnn_model = get_cnn_model()
    history = cnn_model.fit(
        chl_train_x, train_y,
        validation_data=(chl_valid_x, valid_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )
    cnn_model.save(filepath+'/model'+str(model_number))
    cnn_model.load_weights(filepath+'/model'+str(model_number)+'.h5')
    fcst = cnn_model.predict(chl_test_x)
    with open(filepath+'/model'+str(model_number)+'_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    with open(filepath+'/model'+str(model_number)+'_summary.md', 'w') as f:
        with redirect_stdout(f):
            cnn_model.summary()
    return history, fcst, cnn_model


for ld in np.arange(1,24):
    outdr = f"../output/xrchl_t6vRe_hp/models/{ld}"
    predict = []
    # 5번 model ensemble 
    for md in np.arange(5):
        _, fcst, sequence_model = run_experiment(ld, md)
        predict.append(fcst)
    preds = np.array(predict)
    
    np.save(f"{outdr}/fcst{ld}.npy", preds)
 