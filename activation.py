#!/usr/bin/env python
# Generate activation map (based on element-wise activation)
#JHK
import numpy as np
from netCDF4 import Dataset
import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout
import os

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                               Conv2DTranspose, Dense, Dropout,Flatten,
                               GlobalAveragePooling2D, Input, MaxPooling2D,
                               concatenate)
from itertools import chain
# ignore warning message
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#===================================================================================
# gpu setting
#===================================================================================
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    

# main derectory
main_dir = '../output/sst_d35/activation/10/'
os.makedirs('../output/sst_d35/activation/10/', exist_ok=True)
train_path = '/media/cml/Data1/jsp/DLdata/remap/train/'
valid_path = '/media/cml/Data1/jsp/DLdata/remap/valid/'
test_path = '/media/cml/Data1/jsp/DLdata/remap/test/'
 
#===================================================================================
# read test set (GODAS)
#===================================================================================
# input

#shape (11, 21, 72, 36, 3)
# ssh_test = np.load(test_path+'SSH_test_ldmn001-ldmn011.npy')
# chl_test = np.load(test_path+'CHL_test_ldmn001-ldmn011.npy')
sst_test = np.load(test_path+'SST_test_ldmn001-ldmn011.npy')


## Data Cleansing & Pre-Processing

def get_nan_grid(test_npy):
    set_temp = []
    for i in range(test_npy.shape[0]):
        for j in range(test_npy.shape[1]):
            for k in range(test_npy.shape[-1]):
                a = (list(zip(*map(list, np.where(np.isnan(test_npy[i,j,:,:,k])))))) # search nan grid index in single month
                b = (list(set(a))) #unique nan index 
                set_temp.append(b)
    flat = list(chain.from_iterable(set_temp)) #set_temp is 2d list convert to 1d list 
    nan_idx =  list(set(flat)) #unique nan_value grid index 
    return nan_idx

def cleaning(idx, npy):
    for i in range(len(idx)):
        npy[:,:,idx[i][0],idx[i][1],:] = 0 
    return npy 

# chl_idx = get_nan_grid(chl_test)
# print(chl_idx)
# print(len(chl_idx))
# chl_test = cleaning(chl_idx,chl_test)


# ssh_idx = get_nan_grid(ssh_test)
# ssh_test = cleaning(ssh_idx, ssh_test)

sst_idx = get_nan_grid(sst_test)
sst_test = cleaning(sst_idx, sst_test)
sst_test = np.nan_to_num(sst_test)
# chl_test = np.nan_to_num(chl_test)
# ssh_test = np.nan_to_num(ssh_test)

x_data = sst_test[:]
test_x = sst_test[-1,:,:,:]
test_x.shape
test_tdim = sst_test.shape[1]
#===================================================================================
# load model
#===================================================================================

def build_cnn_model():
    inputs = Input(x_data.shape[2:])
    conv1 = Conv2D(35, [3,3], activation='tanh', padding='same', strides=1,kernel_initializer='glorot_normal')(inputs)
    pool1 = MaxPooling2D((2,2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(35, [3,3], activation='tanh', padding='same', strides=1)(pool1)
    pool2 = MaxPooling2D((2,2), strides=2, padding='same')(conv2)
    conv3 = Conv2D(35, [3,3], activation='tanh', padding='same', strides=1)(pool2)
    flat = Flatten()(conv3)
    dense1 = Dense(35, activation='tanh')(flat)
    outputs = Dense(1, activation=None)(dense1)
    return Model(inputs=inputs, outputs=outputs)  

model = load_model('../output/sst_d35/models/10/model2.h5')


#===================================================================================
# activation
#===================================================================================
conv1 = model.layers[1](test_x)
max1 = model.layers[2](conv1)
conv2 = model.layers[3](max1)
max2 = model.layers[4](conv2)

# [tdim,18,9,35]
conv3= np.array(model.layers[5](max2))
# [18*9*35,35] #dense layer's weights 
w1 = np.array(model.layers[7].get_weights()[0])

# [35]
b1 = np.array(model.layers[7].get_weights()[1])

# [35,1]#output layer's weights
w2 = np.array(model.layers[8].get_weights()[0])

# [1]
b2 = np.array(model.layers[8].get_weights()[1])

    #===================================================================================
#===================================================================================
# dimension extension for element-wise muliplication
#===================================================================================
# conv3 [tdim,18,9,35] -> [tdim,18,9,35,35]
conv3 = conv3.reshape(test_tdim, 18, 9, 35, 1)
conv3 = np.repeat(conv3, 35, axis=4)
# w1 [18*6*35,35] -> [tdim,18,9,35,35]
w1 = w1.reshape(1,18,9,35,35)
w1 = np.repeat(w1, test_tdim, axis=0)
# b1 [35] -> [tdim,18,9,35]
b1 = b1.reshape(1,1,1,35)
b1 = np.repeat(b1,test_tdim,axis=0)
b1 = np.repeat(b1,18,axis=1)
b1 = np.repeat(b1,9,axis=2)
b1 /= (18 * 9)
# w2 [35,1] -> [tdim,18,9,35]
w2 = w2.reshape(1,1,1,35)
w2 = np.repeat(w2,test_tdim,axis=0)
w2 = np.repeat(w2,18,axis=1)
w2 = np.repeat(w2,9,axis=2)
# b2 [1] -> [tdim,18,9]
b2 = b2.reshape(1,1,1)
b2 = np.repeat(b2,test_tdim,axis=0)
b2 = np.repeat(b2,18,axis=1)
b2 = np.repeat(b2,9,axis=2)
b2 /= (18 * 9)

#====================================================================================
# generate activation map
#====================================================================================
# FC1 [tdim,18,9,35]
fc1 = np.tanh(np.sum(conv3 * w1, axis=3) + b1) 

# output [tdim,18,9]
activation = np.sum(fc1 * w2, axis=3) + b2

# [tdim,18,9] -> [tdim,6,18]
activation = np.swapaxes(activation, 1, 2)

#===================================================================================
# save
#===================================================================================
activation = np.array(activation)
activation.astype('float32').tofile(main_dir+'activation.gdat')

f = open(main_dir+'activation.ctl','w')
f.write('dset ^activation.gdat\n')
f.write('undef -9.99e+08\n')
f.write('xdef  18  linear   0.  20\n')
f.write('ydef   9  linear -90.  20\n')
f.write('zdef   1  linear 1 1\n')
f.write('tdef '+str(test_tdim)+'  linear jan1998 1yr\n')
f.write('vars   1\n')
f.write('act    1   1  activtion\n')
f.write('ENDVARS\n')
f.close()

os.system('cdo -f nc import_binary '+main_dir+'activation.ctl '+main_dir+'activation.nc')
os.system('rm -f '+main_dir+'activation.ctl '+main_dir+'activation.gdat')
