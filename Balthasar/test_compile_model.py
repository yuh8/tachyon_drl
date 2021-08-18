import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from reinforce import loss_function, caspar_loss_func
from src.CONSTS import MOL_DICT, MAX_MOL_LEN, BATCH_SIZE
from src.embed_utils import import_tf_model


def tokenize_smi(smi):
    N = len(smi)
    i = 0
    token = []
    while i < N:
        for symbol in MOL_DICT:
            if symbol == smi[i:i + len(symbol)]:
                token.append(symbol)
                i += len(symbol)
                break
    return token


def get_encoded_smi(smi):
    tokenized_smi = tokenize_smi(smi)
    encoded_smi = []
    for char in tokenized_smi:
        encoded_smi.append(MOL_DICT.index(char))

    if len(encoded_smi) <= MAX_MOL_LEN:
        num_pads = MAX_MOL_LEN - len(encoded_smi)
        # len(MOL_DICT) is the padding number which will be masked
        encoded_smi += [len(MOL_DICT)] * num_pads
    else:
        encoded_smi = encoded_smi[:MAX_MOL_LEN]
    return encoded_smi


def data_iterator_test(test_df_path):
    df_test = pd.read_csv(test_df_path)
    with open('data/y_max_min.pkl', 'rb') as handle:
        y_min, y_max = pickle.load(handle)
    x = []
    y = []
    for _, row in df_test.iterrows():
        x.append(get_encoded_smi(row.Data))
        _y = (row.pCHEMBL - y_min) / (y_max - y_min)
        y.append(_y)
        if len(x) >= BATCH_SIZE:
            yield (np.vstack(x), np.vstack(y))
            x = []
            y = []

    if x:
        yield (np.vstack(x), np.vstack(y))
        x = []
        y = []


def data_iterator_test_caspar(test_df_path):
    df_test = pd.read_csv(test_df_path)
    x = []
    y = []
    for _, row in df_test.iterrows():
        x.append(get_encoded_smi(row.X))
        y.append(get_encoded_smi(row.Y))
        if len(x) >= BATCH_SIZE:
            yield (np.vstack(x), np.vstack(y))
            x = []
            y = []

    if x:
        yield (np.vstack(x), np.vstack(y))
        x = []
        y = []


def get_optimizer(steps_per_epoch):
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [steps_per_epoch * 10, steps_per_epoch * 20], [0.0001, 0.00001, 0.000001], name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    return opt_op


def compile_caspar_model(melchior):
    melchior.compile(optimizer=get_optimizer(1000), loss=loss_function)
    return melchior


def compile_melchior_model(melchior):
    melchior.compile(optimizer=get_optimizer(1000), loss=tf.keras.losses.MeanSquaredError())
    return melchior


if __name__ == "__main__":
    melchior = import_tf_model("melchior_model")
    # breakpoint()
    melchior_model = compile_melchior_model(melchior)
    # melchior_model.summary()
    breakpoint()

    res = melchior_model.evaluate(data_iterator_test('data/df_test.csv'),
                                  return_dict=True)

    caspar = import_tf_model("pretrained_caspar_model", custom_func=caspar_loss_func)
    caspar_model = compile_caspar_model(caspar)

    res = caspar_model.evaluate(data_iterator_test_caspar('data/df_test_caspar.csv'),
                                return_dict=True)
