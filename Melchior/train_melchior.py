import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from multiprocessing import freeze_support
from data_gen import data_iterator_train, data_iterator_test
from src.embed_utils import (BERTblock,
                             get_token_embedding,
                             get_padding_mask)
from src.misc_utils import create_folder
from src.CONSTS import EMBEDDING_SIZE, MAX_MOL_LEN, NUM_LAYERS, BATCH_SIZE


class MelchiorLayer(layers.Layer):
    def __init__(self, num_layers):
        super(MelchiorLayer, self).__init__()
        self.num_layers = num_layers
        self.bert_layers = [BERTblock() for _ in range(num_layers)]

    def call(self, x, padding_mask):
        for i in range(self.num_layers):
            x = self.bert_layers[i](x, padding_mask)

        #[BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
        return x


def get_melchior_model():
    # [BATCH, MAX_MOL_LEN]
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    token_embedding = get_token_embedding(smi_inputs)
    padding_mask = get_padding_mask(smi_inputs)
    #[BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
    melchior_out = MelchiorLayer(NUM_LAYERS)(token_embedding, padding_mask)
    #[BATCH, EMBEDDING_SIZE]
    # melchior_out = tf.reduce_max(melchior_out, axis=1)
    melchior_out = tf.reshape(melchior_out, (-1, MAX_MOL_LEN * EMBEDDING_SIZE))
    y_pred = layers.Dense(EMBEDDING_SIZE, activation='relu')(melchior_out)
    # [BATCH, 1]
    y_pred = layers.Dense(1, activation=None)(melchior_out)
    return smi_inputs, y_pred


def get_optimizer(steps_per_epoch):
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [steps_per_epoch * 20, steps_per_epoch * 30], [0.0001, 0.00001, 0.000001], name=None
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
    smi_inputs, y_pred = get_melchior_model()
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
