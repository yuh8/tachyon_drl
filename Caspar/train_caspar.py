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


def get_caspar_model():
    # [BATCH, MAX_MOL_LEN]
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    token_embedding = get_token_embedding(smi_inputs)
    padding_mask = get_padding_mask(smi_inputs)
    causal_mask = get_causal_attention_mask()
    caspar_out = CasparLayer(NUM_LAYERS)(token_embedding, padding_mask, causal_mask)
    # [BATCH, MAX_MOL_LEN, DICT_LEN]
    logits = tf.keras.layers.Dense(len(MOL_DICT) + 1)(caspar_out)
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


def get_metrics():
    train_acc = tf.keras.metrics.Accuracy(name="train_acc")
    val_acc = tf.keras.metrics.Accuracy(name="val_acc")
    return train_acc, val_acc


class Caspar(keras.Model):
    def compile(self, optimizer, loss_fn, metric_fn):
        super(Caspar, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_acc, self.val_acc = metric_fn()

    def train_step(self, train_data):
        x, y = train_data

        # capture the scope of gradient
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # compute metrics keeping an moving average
        y_pred_flat = tf.argmax(y_pred, axis=-1)
        self.train_acc.update_state(y, y_pred_flat)
        return {"train_acc": self.train_acc.result()}

    def test_step(self, val_data):
        x, y = val_data
        # predict
        y_pred = self(x, training=False)
        # compute metrics keeping an moving average
        y_pred_flat = tf.argmax(y_pred, axis=-1)
        self.val_acc.update_state(y, y_pred_flat)
        return {"val_acc": self.val_acc.result()}

    @property
    def metrics(self):
        # clear metrics after every epoch
        return [self.train_acc, self.val_acc]


if __name__ == "__main__":
    freeze_support()
    model_path = 'model/train/'
    create_folder(model_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path, save_freq='epoch')]
    steps_per_epoch = pd.read_csv('data/train_data/df_train.csv').shape[0] // BATCH_SIZE
    val_steps = pd.read_csv('data/test_data/df_val.csv').shape[0] // BATCH_SIZE
    # train
    smi_inputs, logits = get_caspar_model()
    model = Caspar(smi_inputs, logits)
    opt_op = get_optimizer()
    model.compile(optimizer=opt_op,
                  loss_fn=loss_func,
                  metric_fn=get_metrics)

    model.fit(data_iterator_train(),
              epochs=30,
              validation_data=data_iterator_test('data/test_data/df_val.csv'),
              validation_steps=val_steps,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)
    res = model.evaluate(data_iterator_test('data/test_data/df_test.csv'),
                         return_dict=True)
