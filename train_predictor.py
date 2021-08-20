import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from multiprocessing import freeze_support
from data_gen_predictor import data_iterator_train, data_iterator_test
from src.embed_utils import get_predictor_model
from src.misc_utils import create_folder
from src.CONSTS import EMBEDDING_SIZE_PRED, BATCH_SIZE_PRED


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model=EMBEDDING_SIZE_PRED, warmup_steps=200):
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


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/predictor/'
    create_folder(ckpt_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq='epoch',
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    mode='min',
                                                    save_best_only=True)]
    steps_per_epoch = pd.read_csv('predictor_data/train_data/df_train.csv').shape[0] // BATCH_SIZE_PRED
    with open('predictor_data/test_data/Xy_val.pkl', 'rb') as handle:
        Xy_val = pickle.load(handle)
    # train
    smi_inputs, y_pred = get_predictor_model()
    opt_op = get_optimizer()
    model = keras.Model(smi_inputs, y_pred)
    model.compile(optimizer='adam',
                  loss='mse')
    model.summary()

    model.fit(data_iterator_train(),
              epochs=60,
              validation_data=Xy_val,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)
    res = model.evaluate(data_iterator_test('predictor_data/test_data/df_test.csv'),
                         return_dict=True)
    model.save_weights('./predictor_weights/predictor')
