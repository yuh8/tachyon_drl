import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from multiprocessing import freeze_support
from data_gen_generator import data_iterator_train, data_iterator_test
from src.embed_utils import get_token_embedding
from src.misc_utils import create_folder, save_model_to_json, load_json_model
from src.CONSTS import MAX_MOL_LEN, MOL_DICT, BATCH_SIZE_GEN, EMBEDDING_SIZE_GEN


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


def get_lstm_model():
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    token_embedding = get_token_embedding(smi_inputs, EMBEDDING_SIZE_GEN)
    padding_mask = get_padding_mask(smi_inputs)
    token_embedding = layers.GRU(256, return_sequences=True, dropout=0.1)(token_embedding, mask=padding_mask)
    token_embedding = layers.GRU(256, return_sequences=True, dropout=0.1)(token_embedding, mask=padding_mask)
    logits = layers.Dense(len(MOL_DICT) + 1)(token_embedding)
    return smi_inputs, logits


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
    smi_inputs, logits = get_lstm_model()
    model = tf.keras.Model(smi_inputs, logits)
    model.compile(optimizer='adam',
                  loss=loss_func)
    model.summary()
    model.fit(data_iterator_train(),
              epochs=2,
              validation_data=Xy_val,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)
    res = model.evaluate(data_iterator_test('generator_data/test_data/df_test.csv'),
                         return_dict=True)

    model.save_weights("./generator_lstm_weights/generator")
    create_folder("generator_lstm_model")
    save_model_to_json(model, "generator_lstm_model/generator_lstm_model.json")
    model_new = load_json_model("generator_lstm_model/generator_lstm_model.json")
    model_new.compile(optimizer='adam',
                      loss=loss_func)
    model_new.load_weights("./generator_lstm_weights/generator")

    res = model_new.evaluate(data_iterator_test('generator_data/test_data/df_test.csv'),
                             return_dict=True)
