import numpy as np
import os
import pickle
from contextlib import redirect_stdout
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


LON_SIZE = 360
LAT_SIZE = 180



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 32
EPOCHS = 135
LEARNING_RATE = 0.005
NUM_FEATURES = 1*3

def get_cnn_model():
    inputs = keras.Input((LON_SIZE, LAT_SIZE, NUM_FEATURES))
    conv1 = keras.layers.Conv2D(35, [3,3], activation='gelu', padding='same', strides=1,kernel_initializer='glorot_normal')(inputs)
    pool1 = keras.layers.MaxPooling2D((2,2), strides=2, padding='same')(conv1)
    conv2 = keras.layers.Conv2D(35, [3,3], activation='gelu', padding='same', strides=1)(pool1)
    pool2 = keras.layers.MaxPooling2D((2,2), strides=2, padding='same')(conv2)
    conv3 = keras.layers.Conv2D(35, [3,3], activation='gelu', padding='same', strides=1)(pool2)
    flat = keras.layers.Flatten()(conv3)
    dense1 = keras.layers.Dense(50, activation='gelu')(flat)
    outputs = keras.layers.Dense(1, activation=None)(dense1)
    cnn_model = keras.Model(inputs=inputs, outputs=outputs) 
    loss = keras.losses.MeanAbsoluteError()
    cnn_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE), loss=loss)
    return cnn_model

#1~66번 LME 지역 연평균 예측 실험 함수 
# Utility for running experiments.
def run_experiment(ldmn=0, model_number=0, lmen=1):
    indir = f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/piConHis/"
    # chl_train_x = np.load(f"{indir}/chl_tr_x_{lmen}.npy")
    # chl_valid_x = np.load(f"{indir}/chl_val_x_{lmen}.npy")
    # chl_test_x = np.load(f"{indir}/chl_test_x_{lmen}.npy")
    
    sst_train_x = np.load(f"{indir}/sst_tr_x.npy")
    val_x1 = np.load(f"{indir}/sst_historical_val_x.npy")
    val_x2 = np.load(f"{indir}/sst_valid_x.npy")
    sst_valid_x = np.concatenate([val_x1,val_x2], axis=0)    
    sst_test_x = np.load(f"{indir}/sst_test_x.npy")

    # train_x = np.load(f"{indir}/tr_x_{lmen}.npy")
    # valid_x = np.load(f"{indir}/val_x_{lmen}.npy")
    # test_x = np.load(f"{indir}/test_x_{lmen}.npy")

    train_y = np.load(f"{indir}/tr_y.npy")
    val_y1 = np.load(f"{indir}/historical_val_y.npy")
    val_y2 = np.load(f"{indir}/valid_y.npy")
    valid_y = np.concatenate([val_y1, val_y2], axis=0)



# 아웃풋 저장 디렉토리 
    outdir = f"/media/cml/Data1/jsp/LMEpredict/gfdl_cm4_sst_pi+his+re/cnn/{ldmn}/{lmen}"
    os.makedirs(outdir, exist_ok=True)       
#텐서보드 로그 저장 디렉토리 
    # logdir= outdir+"/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # os.makedirs(logdir, exist_ok=True)

    filepath = outdir
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"{filepath}/model{model_number}.h5",
            monitor='val_loss',
            save_best_only=True, 
            verbose=1
            ),
# 텐서보드로 학습 추적을 하기 위해 추가한 콜백 
        # tf.keras.callbacks.TensorBoard(
        # log_dir=logdir, 
        # histogram_freq=1,  
        # embeddings_freq=1,  
        # )



    ]
    cnn_model = get_cnn_model()
    history = cnn_model.fit(
        sst_train_x, train_y,
        validation_data=(sst_valid_x, valid_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )
    cnn_model.save(f"{filepath}/model{model_number}.h5")
    cnn_model.load_weights(f"{filepath}/model{model_number}.h5")
    fcst = cnn_model.predict(sst_test_x)
    with open(f"{filepath}/model{model_number}_history", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    with open(f"{filepath}/model{model_number}_summary.md", 'w') as f:
        with redirect_stdout(f):
            cnn_model.summary()
    return history, fcst, cnn_model

for ld in np.arange(1):
    for lme in np.arange(1,67):
        outdr = f'/media/cml/Data1/jsp/LMEpredict/gfdl_cm4_sst_pi+his+re/cnn/{ld}/{lme}'
        predict = []
        for md in np.arange(5):
            _, fcst, sequence_model = run_experiment(ld, md, lme)
            predict.append(fcst)
        preds = np.array(predict)
        np.save(f"{outdr}/fcst{ld}.npy", preds)

