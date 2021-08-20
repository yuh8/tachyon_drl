import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from multiprocessing import freeze_support
from data_gen_generator import data_iterator_train, data_iterator_test
from src.embed_utils import get_generator_model
from src.misc_utils import create_folder
from src.CONSTS import MOL_DICT, BATCH_SIZE_GEN, EMBEDDING_SIZE_GEN


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model=EMBEDDING_SIZE_GEN, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_optimizer():
    opt_op = tf.keras.optimizers.Adam(learning_rate=CustomSchedule())
    return opt_op


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


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator/'
    create_folder(ckpt_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq='epoch',
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    mode='min',
                                                    save_best_only=True)]
    steps_per_epoch = pd.read_csv('generator_data/train_data/df_train.csv').shape[0] // BATCH_SIZE_GEN
    with open('generator_data/test_data/Xy_val.pkl', 'rb') as handle:
        Xy_val = pickle.load(handle)

    # train
    smi_inputs, logits = get_generator_model()
    model = keras.Model(smi_inputs, logits)
    model.compile(optimizer=get_optimizer(),
                  loss=loss_func)
    model.summary()
    model.fit(data_iterator_train(),
              epochs=40,
              validation_data=Xy_val,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)
    res = model.evaluate(data_iterator_test('generator_data/test_data/df_test.csv'),
                         return_dict=True)
    model.save_weights('./weights/generator')
