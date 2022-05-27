

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from contextlib import redirect_stdout

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import kerastuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, optimizers
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

LON_SIZE = 360
LAT_SIZE = 180

NUM_FEATURES = 1*3

# 하이퍼파리미터를 찾기 위해 모델에서 조정할 레이어의 값 범위를 지정해주고 추가로 학습률과 배치 크기의 값 범위를 추가로 지정해줍니다.
class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    hp_units1 = hp.Int('units_1', min_value = 16, max_value = 180)
    hp_units2 = hp.Int('units_2', min_value = 16, max_value = 180)
    hp_units4 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.05)
    hp_units5 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.05)

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
    loss = keras.losses.MeanAbsoluteError()
    model.compile(optimizer=opt, loss=loss, metrics=[keras.metrics.RootMeanSquaredError(name='my_rmse')])
    return model

  def fit(self, hp, model, *args, **kwargs):
      return model.fit(
          *args,
          batch_size=hp.Int('batch_size', min_value = 16, max_value = 256, step = 16),
          **kwargs,
      )

for lme in np.arange(1,67):
    outdr = f'/media/cml/Data1/jsp/LMEpredict/xrsst_base/cnn/0/{lme}'
    predict = []
    indir =  f"/media/cml/Data1/jsp/LMEdata/0/{lme}/RGB/"
    for model_number in np.arange(5):

        # chl_train_x = np.load(f"{indir}/chl_tr_x_{lme}.npy")
        # chl_valid_x = np.load(f"{indir}/chl_val_x_{lme}.npy")
        # chl_test_x = np.load(f"{indir}/chl_test_x_{lme}.npy")

        sst_train_x = np.load(f"{indir}/sst_tr_x_{lme}.npy")
        sst_valid_x = np.load(f"{indir}/sst_val_x_{lme}.npy")
        sst_test_x = np.load(f"{indir}/sst_test_x_{lme}.npy")
        # train_x = np.load(f"{indir}/tr_x_{lme}.npy")
        # valid_x = np.load(f"{indir}/val_x_{lme}.npy")
        # test_x = np.load(f"{indir}/test_x_{lme}.npy")
        train_y = np.load(f"{indir}/tr_y_{lme}.npy")
        valid_y = np.load(f"{indir}/val_y_{lme}.npy")



        # 최적 하이퍼파라미터를 찾는 작업을 진행합니다.
        tuner = kt.Hyperband(MyHyperModel(),
                            objective='val_loss',
                            max_epochs = 100,
                            executions_per_trial = 3,
                            overwrite = True,
                            factor = 3)

        tuner.search(sst_train_x, train_y,
                validation_data=(sst_valid_x, valid_y), 
                epochs = 100)



        # 최적 하이퍼파라미터를 가져옵니다.
        best_hp = tuner.get_best_hyperparameters()[0]



        # 최적 하이퍼파라미터를 출력합니다.

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
        batch_size : {best_hp.get('batch_size')}
        """)


        # 배치 크기는 따로 저장했다가 fit 메소드에서 적용합니다.
        batch_size = best_hp.get('batch_size')

        outdir = f"/media/cml/Data1/jsp/LMEpredict/xrsst_base/cnn/0/{lme}"
        os.makedirs(outdir, exist_ok=True) 
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
                ),]
        # 최적값으로 모델을 생성합니다.
        model = tuner.hypermodel.build(best_hp)


        # 학습을 진행합니다.
        # validation_split 아규먼트로 Training 데이터셋의 10%를 Validation 데이터셋으로 사용하도록합니다.
        # 예를 들어 배치 크기가 256이라는 것은 전체 데이터셋을 샘플 256개씩으로 나누어서 학습에 사용한다는 의미입니다.
        # 예를 들어 에포크(epochs)가 10이라는 것은 전체 train 데이터셋을 10번 본다는 의미입니다.
        history = model.fit(
            sst_train_x, train_y,
            validation_data=(sst_valid_x, valid_y),
        batch_size=batch_size,
        epochs=135,
        )

        model.save(f"{filepath}/model{model_number}.h5")
        model.load_weights(f"{filepath}/model{model_number}.h5")
        fcst = model.predict(sst_test_x)
        with open(f"{filepath}/model{model_number}_history", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        with open(f"{filepath}/model{model_number}_summary.md", 'w') as f:
            with redirect_stdout(f):
                model.summary()
    
        predict.append(fcst)
    preds = np.array(predict)
    np.save(f"{outdr}/fcst0.npy", preds)

