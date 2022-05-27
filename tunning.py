import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LON_SIZE = 360
LAT_SIZE = 180


LD = 0
LME = 30
NUM_FEATURES = 1*3

BATCH_SIZE = 32
EPOCHS = 50

ldmn = LD
lmen = LME 

indir =  f"/media/cml/Data1/jsp/LMEdata/{ldmn}/{lmen}/RGB/"


# chl_train_x = np.load(f"{indir}/chl_tr_x_{lmen}.npy")
# chl_valid_x = np.load(f"{indir}/chl_val_x_{lmen}.npy")
# chl_test_x = np.load(f"{indir}/chl_test_x_{lmen}.npy")

sst_train_x = np.load(f"{indir}/sst_tr_x_{lmen}.npy")
sst_valid_x = np.load(f"{indir}/sst_val_x_{lmen}.npy")
sst_test_x = np.load(f"{indir}/sst_test_x_{lmen}.npy")
# train_x = np.load(f"{indir}/tr_x_{lmen}.npy")
# valid_x = np.load(f"{indir}/val_x_{lmen}.npy")
# test_x = np.load(f"{indir}/test_x_{lmen}.npy")
train_y = np.load(f"{indir}/tr_y_{lmen}.npy")
valid_y = np.load(f"{indir}/val_y_{lmen}.npy")


from tensorflow import keras
from tensorflow.keras import layers, backend
import keras_tuner as kt
import tensorflow as tf

def plcc_metric(y_true, y_pred):  
    x = y_true
    y = y_pred
    mx = backend.mean(x)
    my = backend.mean(y)
    xm, ym = x-mx, y-my
    r_num = backend.sum(tf.multiply(xm,ym))
    r_den = backend.sqrt(tf.multiply(backend.sum(backend.square(xm)), backend.sum(backend.square(ym)))) + 1e-12
    return r_num / r_den


def build_model(hp):
    """
    Build model for hyperparameters tuning
    
    hp: HyperParameters class instance
    """
    model = keras.Sequential()
    inputShape = (LON_SIZE, LAT_SIZE, NUM_FEATURES)
    model.add(keras.layers.Conv2D(
		hp.Int("conv_1", min_value=32, max_value=96, step=32),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5,7]), 
        padding="same", kernel_initializer='glorot_normal',input_shape=inputShape))
    model.add(keras.layers.Activation("gelu"))
    # model.add(keras.layers.BatchNormalization(axis=chanDim))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	# second CONV => TANH => POOL layer set
    model.add(keras.layers.Conv2D(
		hp.Int("conv_2", min_value=64, max_value=128, step=32),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5,7]), 
        padding="same"))
    model.add(keras.layers.Activation("gelu"))
    # model.add(keras.layers.BatchNormalization(axis=chanDim))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # second CONV => TANH => Dense layer set
    model.add(keras.layers.Conv2D(
		hp.Int("conv_3", min_value=64, max_value=128, step=32),
        kernel_size=hp.Choice('conv_3_kernel', values = [3,5,7]), 
        padding="same"))    
    model.add(keras.layers.Flatten())
    hp_unit = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_unit, activation="gelu"))
    model.add(keras.layers.Dense(1, activation=None))
    hp_learnig_rate = hp.Choice('learning_rate',values = [1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4])
    opt = keras.optimizers.Adagrad(learning_rate=hp_learnig_rate)
    # opt =  hp.Choice('optimizer', values=['adagrad', 'rmsprop', 'adam'])
    # if opt == 'adagrad':
    #     optimizer = keras.optimizers.SGDc
    # elif opt == 'rmsprop':
    #     optimizer = keras.optimizers.RMSprop(learning_rate=hp_learnig_rate)
    # elif opt == 'adam':
    #     optimizer = keras.optimizers.Adam(learning_rate=hp_learnig_rate)
    # else:
    #     raise
    loss = keras.losses.MeanAbsoluteError()
    model.compile(optimizer= opt, loss=loss, metrics=[plcc_metric,keras.metrics.RootMeanSquaredError(name='my_rmse')])

    return model

tuner = kt.Hyperband(build_model,
                    objective='val_loss',
                    # objective=kt.Objective("my_mse", direction="min"),
                    max_epochs=50,
                    factor=3,
                    directory = '/media/cml/Data1/jsp/LMEpredict/xrsst_Gelu_norm/tunning/',
                    project_name='mae_loss')

tuner.search(sst_train_x, train_y,
        validation_data=(sst_valid_x, valid_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS)
print(tuner.search_space_summary())
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
optimized conv_1_filter number : {best_hp.get("conv_1")} 
optimized conv_1_kernel size : {best_hp.get('conv_1_kernel')} 
optimized conv_2_filter number : {best_hp.get("conv_2")} 
optimized conv_2_kernel size : {best_hp.get('conv_2_kernel')}
optimized conv_3_filter number : {best_hp.get("conv_3")} 
optimized conv_3_kernel size : {best_hp.get('conv_3_kernel')}
optimized dense_1_units number: {best_hp.get('units')}
optimizer : {best_hp.get('optimizer')} 
learnig rate : {best_hp.get('learning_rate')} 
""")