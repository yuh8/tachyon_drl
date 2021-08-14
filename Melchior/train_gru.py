import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from multiprocessing import freeze_support
from data_gen import data_iterator_train, data_iterator_test
from src.misc_utils import create_folder
from src.CONSTS import (MAX_MOL_LEN, MOL_DICT,
                        NUM_LAYERS, BATCH_SIZE)


def get_padding_mask(x):
    # [BATCH, MAX_MOL_LEN]
    padding_mask = tf.less(x, len(MOL_DICT))
    return padding_mask


class GRULayer(layers.Layer):
    def __init__(self):
        super(GRULayer, self).__init__()
        self.embedding = layers.Embedding(len(MOL_DICT) + 1, 128)
        self.gru1 = layers.GRU(128, return_sequences=True, dropout=0.3)
        self.gru2 = layers.GRU(128, dropout=0.3)
        self.dense = layers.Dense(128, activation='relu')

    def call(self, x, padding_mask):
        x = self.embedding(x)
        x = self.gru1(x, mask=padding_mask)
        x = self.gru2(x, mask=padding_mask)
        #[BATCH, MAX_MOL_LEN, 128]
        y_pred = self.dense(x)
        return y_pred


def get_gru_model():
    # [BATCH, MAX_MOL_LEN]
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    padding_mask = get_padding_mask(smi_inputs)
    gru_out = GRULayer()(smi_inputs, padding_mask)
    # [BATCH, MAX_MOL_LEN, DICT_LEN]
    y_pred = layers.Dense(1, activation=None)(gru_out)
    return smi_inputs, y_pred


def get_optimizer(steps_per_epoch):
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [steps_per_epoch * 20, steps_per_epoch * 30], [0.001, 0.0001, 0.00001], name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    return opt_op


if __name__ == "__main__":
    freeze_support()
    model_path = 'model/train/'
    create_folder(model_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path,
                                                    save_freq='epoch',
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    mode='min',
                                                    save_best_only=True)]
    steps_per_epoch = pd.read_csv('data/train_data/df_train.csv').shape[0] // BATCH_SIZE
    with open('data/test_data/Xy_val.pkl', 'rb') as handle:
        Xy_val = pickle.load(handle)
    # train
    smi_inputs, y_pred = get_gru_model()
    opt_op = get_optimizer(steps_per_epoch)
    model = keras.Model(smi_inputs, y_pred)
    model.compile(optimizer='adam',
                  loss='mse')
    model.summary()

    model.fit(data_iterator_train(),
              epochs=60,
              validation_data=Xy_val,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)
    res = model.evaluate(data_iterator_test('data/test_data/df_test.csv'),
                         return_dict=True)
    model.save('model/Melchior/', save_traces=False)
