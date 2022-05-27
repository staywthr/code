import glob
import os
import pickle
import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
from contextlib import redirect_stdout




LOCS = np.load('/media/cml/Data1/jsp/DATA/1/test/chl_latlon_mask_1x1.npy')

LON_SIZE = 360
LAT_SIZE = 180

###train
train_path = '/media/cml/Data1/jsp/DATA/cmip6/'
chl_train_files = sorted(glob.glob(train_path + 'sfc_chl_Omon_*.nc'))
sst_train_files = sorted(glob.glob(train_path + 'tos_Omon_*.nc'))
target_train_files = sorted(glob.glob(train_path + 'target/tos_Omon_*_target.nc'))

print(f"Total cmip6 models(chl) for training: {len(chl_train_files)}")
print(f"Total cmip6 models(sst) for training: {len(sst_train_files)}")
print(f"Total cmip6 models(target) for training: {len(target_train_files)}")

####valid
valid_path = "/media/cml/Data1/jsp/DATA/1/transfer/"
chl_valid_files = sorted(glob.glob(valid_path+"sfc_chl_*196501*.nc"))
sst_valid_files = sorted(glob.glob(valid_path+"sst_*196501*.nc"))
ssh_valid_files =  sorted(glob.glob(valid_path+"eta_t_*196501*.nc"))
target_valid_files = sorted(glob.glob(valid_path+"sst_*196501*_target.nc"))


test_path = "/media/cml/Data1/jsp/DATA/1/test/"
chl_test_files = sorted(glob.glob(test_path+"chl_199801-201812.nc"))
sst_test_files = sorted(glob.glob(test_path+"sst_JAN1998*.nc"))
ssh_test_files =  sorted(glob.glob(test_path+"ssh_JAN1998*.nc"))
target_test_files = sorted(glob.glob(test_path+"sst_JAN1998*_target.nc"))

def chk_var(klist): 
    var_name_list = ['eta_t','sshg','ssh','SSH','chl','sfc_chl','chlos','tos','sst','SST']
    var = list(set(klist).intersection(var_name_list))
    return var

def load_xdata(path):
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    arr = data[v[0]].sel()
    dim = list(arr.dims)
    shape =  ['time','lon','lat']
    if v == ['chl'] and len(arr.dims) == 4:
        arr = arr.squeeze(list(set(dim).difference(set(shape))))
    return arr.transpose('time','lon','lat')




def cleanining_grid(da, locnpy):
    for i in range(len(locnpy)):
        da[:,locnpy[i][0],locnpy[i][1]] = np.nan 
    print(f'# Total number of missing grid: {len(locnpy)}')
    return da

def calculate_anomaly(da, groupby_type="time.month"):
    gb = da.groupby(groupby_type)
    clim = gb.mean(dim="time")
    return gb - clim


def calculate_stand_anomaly(da, groupby_type="time.month"):
    gb = da.groupby(groupby_type)
    clim = gb.mean(dim="time")
    clim_s =gb.std(dim="time")
    stand_anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        gb,
        clim,
        clim_s,
    )
    return stand_anomalies


def month_anomaly(var_anom, mind):
    xda = var_anom.where(var_anom.coords['month']==mind,drop=True)
    return xda


def season_anomaly(var_anom,ld):
    """
    returns 3 month anomaly xarray.DataArray.
    Parameters
    ------------------------
    ld = lead month
    """
    if ld == 1:
        anom0 = month_anomaly(var_anom, 11)
        anom1 = month_anomaly(var_anom, 12)
        anom2 = month_anomaly(var_anom, 1)
        ldmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)),anom2.isel(time=slice(1, None))], dim='z', join='override')
    elif ld == 12:
        anom0 = month_anomaly(var_anom, 12)
        anom1 = month_anomaly(var_anom, 1)
        anom2 = month_anomaly(var_anom, 2)
        ldmn = xr.concat([anom0.isel(time=slice(0, -2)), anom1.isel(time=slice(1, -1)),anom2.isel(time=slice(1, -1))], dim='z', join='override')
    elif ld == 13:
        anom0 = month_anomaly(var_anom, 11)
        anom1 = month_anomaly(var_anom, 12)
        anom2 = month_anomaly(var_anom, 1)
        ldmn = xr.concat([anom0.isel(time=slice(0, -2)), anom1.isel(time=slice(0, -2)),anom2.isel(time=slice(1, -1))], dim='z', join='override')
    elif ld >13:
        anom0 = month_anomaly(var_anom, 25-ld-1)
        anom1 = month_anomaly(var_anom, 25-ld)
        anom2 = month_anomaly(var_anom, 25-ld+1)
        ldmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)), anom2.isel(time=slice(0, -1))], dim='z', join='override')
    else:
        anom0 = month_anomaly(var_anom, 13-ld-1)
        anom1 = month_anomaly(var_anom, 13-ld)
        anom2 = month_anomaly(var_anom, 13-ld+1)
        ldmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)), anom2.isel(time=slice(0, None))], dim='z', join='override')
    return ldmn.transpose('time','lon','lat','z')

#xarray 
def log_transform(arr):
    epsilon=1e-06 
    log_norm = np.log(arr+epsilon)
    return log_norm

#xarray 
def log10_transform(arr):
    epsilon=1e-06 
    log_norm = np.log10(arr+epsilon)
    return log_norm

def prepare_single_frame(path, ld): 
    """
    returns lead months single feature(chl/sst/ssh) xarray from one file
    Parameters
    ------------------------
    path = file path
    ld = lead month
    """
    arr = load_xdata(path)
    if (arr.name == 'chl') or (arr.name == 'chlos') or (arr.name == 'sfc_chl'):
        print("chlorophyll preprocess")
        arr = cleanining_grid(arr, LOCS)
        arr = log_transform(arr)
    
    anom = calculate_anomaly(arr)
    # anom = calculate_stand_anomaly(arr)
    ss_anom = season_anomaly(anom, ld)
   
    print(f"Lead month {ld} {ss_anom.name} {(path.partition('Omon_')[-1])} Dataset")
    return ss_anom


# ldxds = xr.Dataset({ss_anom.name : ss_anom})
# xds.merge(xds2)

def sequence_generator(offset, chunk, xds):
    batch_ds = []
    for i in range(int(len(xds['time'])//chunk)):
        batch_ds.append(xds.isel(time=slice(offset, offset+chunk)))
        offset+=chunk
    return batch_ds

def label_generator(path, ld):
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    t_arr = data[v[0]].sel()
    epsilon=1e-06 
    if ld==1:
        label = t_arr.values #164
    elif ld==12:
        label = t_arr.values[1:] #163
    elif ld ==13:
        label = t_arr.values[1:] #163
    elif ld>13:
        label =  np.append(t_arr.values[1:], [0+epsilon]) #164
    else: # 0<ld<12
        label =  np.append(t_arr.values, [0+epsilon]) #165
    return label

def prepare_ld_sequence(paths, ld):
    """
    returns lead months single feature(chl/sst/ssh) xarray and nino indexes from files
    Parameters
    ------------------------
    path = files path
    ld = lead month
    """
    temp_frames = [] #lead months xarray list 
    temp_labels = [] #lead months nino index xarray list 
    # For each video.
    for path in (paths):
        # Gather all its frames and add a batch dimension.
        frames = prepare_single_frame(path[0],ld)
        label = label_generator(path[1], ld)
        temp_frames.append(frames)
        temp_labels.append(label)
    # frame_features = [el.interpolate_na(dim="lon") for el in temp_frames]    
    frame_features = [el.fillna(0) for el in temp_frames]
    sequence = np.concatenate(frame_features, axis=0)
    labels = np.concatenate(temp_labels, axis=0)
    labels = labels[:,np.newaxis]
    return sequence , labels





# ss_anom.rolling(time=3, center=True).mean()[1::3]

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
    chl_train_paths = list(zip(chl_train_files, target_train_files))
    chl_train_x, train_y = prepare_ld_sequence(chl_train_paths, ldmn)
    chl_valid_paths = list(zip(chl_valid_files, target_valid_files))
    chl_valid_x, valid_y = prepare_ld_sequence(chl_valid_paths, ldmn)
    chl_test_paths = list(zip(chl_test_files, target_test_files))
    chl_test_x, _ = prepare_ld_sequence(chl_test_paths, ldmn)

    # sst_train_paths = list(zip(sst_train_files, target_train_files))
    # sst_train_x, _ = prepare_ld_sequence(sst_train_paths, ldmn)
    # sst_valid_paths = list(zip(sst_valid_files, target_valid_files))
    # sst_valid_x, _ = prepare_ld_sequence(sst_valid_paths, ldmn)
    # sst_test_paths = list(zip(sst_test_files, target_test_files))
    # sst_test_x, _ = prepare_ld_sequence(sst_test_paths, ldmn)

    # train_x = np.append(chl_train_x, sst_train_x, axis=3)
    # valid_x = np.append(chl_valid_x, sst_valid_x, axis=3)
    # test_x = np.append(chl_test_x, sst_test_x, axis=3)

    os.makedirs('../output/xrchl_t6vRe_hp/models/'+str(ldmn), exist_ok=True)
    filepath = '../output/xrchl_t6vRe_hp/models/'+str(ldmn)
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
    predict = []
    for md in np.arange(5):
        _, fcst, sequence_model = run_experiment(ld, md)
        predict.append(fcst)
    preds = np.array(predict)
    np.save('../output/xrchl_t6vRe_hp/models/'+str(ld)+'/fcst'+str(ld)+'.npy', preds)
 
