import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from multiprocessing import freeze_support
from data_gen import data_iterator_train, data_iterator_test
from src.misc_utils import create_folder
from src.CONSTS import (EMBEDDING_SIZE, MAX_MOL_LEN, MOL_DICT, BATCH_SIZE)


def get_padding_mask(x):
    # [BATCH, MAX_MOL_LEN]
    padding_mask = tf.less(x, len(MOL_DICT))
    return padding_mask


def loss_func(y, logits):
    '''
    y : [BATCH, MAX_MOL_LEN]
    logits: [BATCH, MAX_MOL_LEN, DICT_SIZE]
    '''
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.less(y, len(MOL_DICT))
    _loss = loss_obj(y, logits)
    mask = tf.cast(mask, _loss.dtype)
    _loss *= mask
    return tf.reduce_sum(_loss) / tf.reduce_sum(mask)


class CasparLayer(layers.Layer):
    def __init__(self):
        super(CasparLayer, self).__init__()
        self.embedding = layers.Embedding(len(MOL_DICT) + 1, EMBEDDING_SIZE)
        self.gru = layers.GRU(1024, return_sequences=True, dropout=0.3)
        self.dense = layers.Dense(len(MOL_DICT) + 1)

    def call(self, x, padding_mask):
        x = self.embedding(x)
        x = self.gru(x, mask=padding_mask)
        #[BATCH, MAX_MOL_LEN, MOL_DICT_LEN+1]
        logits = self.dense(x)
        return logits


def get_caspar_model():
    # [BATCH, MAX_MOL_LEN]
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    padding_mask = get_padding_mask(smi_inputs)
    caspar_out = CasparLayer()(smi_inputs, padding_mask)
    # [BATCH, MAX_MOL_LEN, DICT_LEN]
    logits = layers.Dense(len(MOL_DICT) + 1)(caspar_out)
    return smi_inputs, logits


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
    smi_inputs, logits = get_caspar_model()
    model = keras.Model(smi_inputs, logits)
    model.compile(optimizer='adam',
                  loss=loss_func)
    model.summary()

    model.fit(data_iterator_train(),
              epochs=30,
              validation_data=Xy_val,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)
    res = model.evaluate(data_iterator_test('data/test_data/df_test.csv'),
                         return_dict=True)
    model.save('model/Caspar/', save_traces=False)