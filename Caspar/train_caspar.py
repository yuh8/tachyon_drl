import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from multiprocessing import freeze_support
from data_gen import data_iterator_train, data_iterator_test
from src.embed_utils import (GPTBlock, get_token_embedding,
                             get_padding_mask, get_causal_attention_mask)
from src.misc_utils import create_folder
from src.CONSTS import (EMBEDDING_SIZE, MAX_MOL_LEN,
                        NUM_LAYERS, MOL_DICT, BATCH_SIZE)


class CasparLayer(layers.Layer):
    def __init__(self, num_layers):
        super(CasparLayer, self).__init__()
        self.num_layers = num_layers
        self.gpt_layers = [GPTBlock() for _ in range(num_layers)]

    def call(self, x, padding_mask, causal_mask):
        for i in range(self.num_layers):
            x = self.gpt_layers[i](x, padding_mask, causal_mask)

        #[BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
        return x


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.embed_size = tf.cast(EMBEDDING_SIZE, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embed_size) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {'warmup_steps': self.warmup_steps}
        return config


def get_caspar_model():
    # [BATCH, MAX_MOL_LEN]
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    token_embedding = get_token_embedding(smi_inputs)
    padding_mask = get_padding_mask(smi_inputs)
    causal_mask = get_causal_attention_mask()
    caspar_out = CasparLayer(NUM_LAYERS)(token_embedding, padding_mask, causal_mask)
    # [BATCH, MAX_MOL_LEN, DICT_LEN]
    logits = layers.Dense(len(MOL_DICT) + 1)(caspar_out)
    return smi_inputs, logits


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


def get_optimizer():
    lr_fn = CustomSchedule()
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn,
                                      beta_1=0.9,
                                      beta_2=0.98,
                                      epsilon=1e-9)
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
    smi_inputs, logits = get_caspar_model()
    model = keras.Model(smi_inputs, logits)
    model.compile(optimizer=get_optimizer(),
                  loss=loss_func)
    model.summary()

    model.fit(data_iterator_train(),
              epochs=3,
              validation_data=Xy_val,
              callbacks=callbacks,
              steps_per_epoch=100)
    res = model.evaluate(data_iterator_test('data/test_data/df_test.csv'),
                         return_dict=True)
    model.save('model/Caspar/', save_traces=False)
