import numpy as np
import os
from datetime import datetime
import pickle
import tensorflow as tf
from contextlib import redirect_stdout
from tensorflow import keras
from tensorflow.keras import layers
#conda install -c esri tensorflow-addons
import tensorflow_addons as tfa

try:  # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:  # detect GPUs
    tpu = False
    strategy = (
        tf.distribute.get_strategy()
    )  # default strategy that works on CPU and single GPU
print("Number of Accelerators: ", strategy.num_replicas_in_sync)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LON_SIZE = 360
LAT_SIZE = 180

BATCH_SIZE = 8
EPOCHS = 135
NUM_FEATURES = 1*3
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.05


#LR Scheduler Utility
# Reference:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
class StochasticDepth(layers.Layer):
    """Stochastic Depth module.
    It is also referred to as Drop Path in `timm`.
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class Block(tf.keras.Model):
    """ConvNeXt block.
    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.dim = dim
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None
        # Depthwise Convolution(Swin-T의 MSA)
        self.dw_conv_1 = layers.Conv2D(
            filters=dim, kernel_size=7, padding="same", groups=dim
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        # Pointwise Convolution(Swin-T의 MLP)
        self.pw_conv_1 = layers.Dense(4 * dim)
        self.act_fn = layers.Activation("gelu")
        self.pw_conv_2 = layers.Dense(dim)
        # stochastic depth
        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )

    def call(self, inputs):
        x = inputs

        x = self.dw_conv_1(x)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x) # (N, H, W, C*4)
        x = self.act_fn(x)
        x = self.pw_conv_2(x)  # (N, H, W, C)

        if self.gamma is not None:
            x = self.gamma * x

        return inputs + self.drop_path(x)  # skip connection


def get_convnext_model(
    model_name="convnext_tiny_1k",
    input_shape=(LON_SIZE, LAT_SIZE, NUM_FEATURES),
    num_classes=1,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
) -> keras.Model:
    """Implements ConvNeXt family of models given a configuration.
    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Note: `predict()` fails on CPUs because of group convolutions. The fix is recent at
    the time of the development: https://github.com/keras-team/keras/pull/15868. It's
    recommended to use a GPU / TPU.
    """

    inputs = layers.Input(input_shape)
    stem = keras.Sequential(
        [
            layers.Conv2D(dims[0], kernel_size=4, strides=4), # like patch embedding
            layers.LayerNormalization(epsilon=1e-6),
        ],
        name="stem",
    )

    downsample_layers = []
    downsample_layers.append(stem)
    for i in range(3):
        downsample_layer = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6), # 발산을 막기 위한 normalize
                layers.Conv2D(dims[i + 1], kernel_size=2, strides=2),
            ],
            name=f"downsampling_block_{i}",
        )
        downsample_layers.append(downsample_layer)     

    stages = []
    dp_rates = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]  # stage가 깊어질수록 stochastic depth가 적용될 확률이 높아짐
    cur = 0
    for i in range(4):
        stage = keras.Sequential(
            [
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        name=f"convnext_block_{i}_{j}",
                    )
                    for j in range(depths[i])
                ]
            ],
            name=f"convnext_stage_{i}",
        )
        stages.append(stage)
        cur += depths[i] # next stage

    x = inputs
    for i in range(len(stages)):
        x = downsample_layers[i](x) # stem, downsample_layer1, downsample_layer2, downsample_layer3
        x = stages[i](x) # stage1, stage2, stage3, stage4

    x = layers.GlobalAvgPool2D()(x)  # global average pooling, (N, C, H, W) -> (N, C)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(50, activation='gelu')(x)
    outputs = layers.Dense(num_classes, name="regression_head")(x)
    cnn_model = keras.Model(inputs, outputs, name=model_name)
    loss = keras.losses.MeanAbsoluteError()
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
    optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    cnn_model.compile(optimizer=optimizer, loss=loss)
    return cnn_model

# conda install -c esri tensorflow-addons

# EPOCHS = 10
# WARMUP_STEPS = 10
# INIT_LR = 0.03
# WAMRUP_LR = 0.006

# TOTAL_STEPS = int((2460/ BATCH_SIZE) * EPOCHS)

# scheduled_lrs = WarmUpCosine(
#     learning_rate_base=INIT_LR,
#     total_steps=TOTAL_STEPS,
#     warmup_learning_rate=WAMRUP_LR,
#     warmup_steps=WARMUP_STEPS,
# )
# import matplotlib.pyplot as plt
# lrs = [scheduled_lrs(step) for step in range(TOTAL_STEPS)]
# plt.figure(figsize=(10, 6))
# plt.plot(lrs)
# plt.xlabel("Step", fontsize=14)
# plt.ylabel("LR", fontsize=14)
# plt.show()
# optimizer = keras.optimizers.SGD(scheduled_lrs)
# loss = keras.losses.MeanAbsoluteError()

# # Utility for running experiments.
def run_experiment(ldmn=0, model_number=0, lmen=1):
    indir = f"/media/cmlws/Data2/jsp/LMEdata/{ldmn}/{lmen}/RGB/"
    # chl_train_x = np.load(f"{indir}/chl_tr_x_{lmen}.npy")
    # chl_valid_x = np.load(f"{indir}/chl_val_x_{lmen}.npy")
    # chl_test_x = np.load(f"{indir}/chl_test_x_{lmen}.npy")
    
    # sst_train_x = np.load(f"{indir}/sst_tr_x_{lmen}.npy")
    # sst_valid_x = np.load(f"{indir}/sst_val_x_{lmen}.npy")
    # sst_test_x = np.load(f"{indir}/sst_test_x_{lmen}.npy")

    train_x = np.load(f"{indir}/chl_tr_x_{lmen}.npy")
    valid_x = np.load(f"{indir}/chl_val_x_{lmen}.npy")
    test_x = np.load(f"{indir}/chl_test_x_{lmen}.npy")

    train_y = np.load(f"{indir}/tr_y_{lmen}.npy")
    valid_y = np.load(f"{indir}/val_y_{lmen}.npy")

    outdir = f"/media/cmlws/Data2/jsp/LMEpredict/xrcs_t6vRe_norm/conv_next/{ldmn}/{lmen}"
    # os.makedirs(outdir, exist_ok=True)       
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
        
        # tf.keras.callbacks.TensorBoard(
        # log_dir=logdir, 
        # histogram_freq=1,  
        # embeddings_freq=1,  
        # )



    ]
    cnn_model = get_convnext_model()
    history = cnn_model.fit(
        train_x, train_y,
        validation_data=(valid_x, valid_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )

    cnn_model.save(f"{filepath}/model{model_number}.h5")
    cnn_model.load_weights(f"{filepath}/model{model_number}.h5")
    fcst = cnn_model.predict(test_x)
    with open(f"{filepath}/model{model_number}_history", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    with open(f"{filepath}/model{model_number}_summary.md", 'w') as f:
        with redirect_stdout(f):
            cnn_model.summary()
    return history, fcst, cnn_model

for ld in np.arange(1):
    for lme in np.arange(1,67):
        outdr = f'/media/cmlws/Data2/jsp/LMEpredict/xrcs_t6vRe_mean/conv_next/{ld}/{lme}'
        predict = []
        for md in np.arange(5):
            _, fcst, sequence_model = run_experiment(ld, md, lme)
            predict.append(fcst)
        preds = np.array(predict)
        np.save(f"{outdr}/fcst{ld}.npy", preds)
